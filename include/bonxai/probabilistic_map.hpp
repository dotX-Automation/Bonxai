#pragma once

#include <eigen3/Eigen/Geometry>
#include <unordered_set>
#include <bonxai/bonxai.hpp>

namespace Bonxai
{

template<class Functor>
void RayIterator(
  const CoordT & key_origin,
  const CoordT & key_end,
  const double prob_miss,
  const Functor & func);

inline void ComputeRay(
  const CoordT & key_origin,
  const CoordT & key_end,
  const double prob_miss,
  std::vector<CoordT> & ray)
{
  ray.clear();
  RayIterator(
    key_origin, key_end, prob_miss, [&ray](const CoordT & coord, const double prob_miss)
    {
      (void)prob_miss;
      ray.push_back(coord);
      return true;
    });
}

/**
 * @brief The ProbabilisticMap class is meant to behave as much as possible as
 * octomap::Octree, given the same voxel size.
 *
 * Insert a point cloud to update the current probability
 */
class ProbabilisticMap
{
public:
  using Vector3D = Eigen::Vector3d;

  /// Compute the logds, but return the result as an integer,
  /// The real number is represented as a fixed precision
  /// integer (6 decimals after the comma)
  [[nodiscard]] static constexpr int32_t logods(float prob)
  {
    return int32_t(1e6 * std::log(prob / (1.0 - prob)));
  }

  /// Expect the fixed comma value returned by logods()
  [[nodiscard]] static constexpr float prob(int32_t logods_fixed)
  {
    float logods = float(logods_fixed) * 1e-6;
    return 1.0 - 1.0 / (1.0 + std::exp(logods));
  }

  struct CellT
  {
    // variable used to check if a cell was already updated in this loop
    int32_t update_id : 4;
    // the probability of the cell to be occupied
    int32_t probability_log : 28;

    CellT()
    : update_id(0)
      , probability_log(UnknownProbability) {}
  };

  static const int32_t UnknownProbability;

  ProbabilisticMap(double resolution);

  [[nodiscard]] VoxelGrid<CellT> & grid();

  [[nodiscard]] const VoxelGrid<CellT> & grid() const;

  void setThresMin(const double thres_min);

  double getThresMin();

  void setThresMax(const double thres_max);

  double getThresMax();

  void setThresOccupancy(const double thres_occupancy);

  /**
   * @brief insertPointCloud will update the probability map
   * with a new set of detections.
   * The template function can accept points of different types,
   * such as pcl:Point, Eigen::Vector or Bonxai::Point3d
   *
   * Both origin and points must be in world coordinates
   *
   * @param points   a vector of points which represent detected obstacles
   * @param origin   origin of the point cloud
   * @param max_range  max range of the ray, if exceeded, we will use that
   * to compute a free space
   */
  template<typename PointT, typename Allocator>
  void insertPointCloud(
    const std::vector<PointT, Allocator> & points,
    const PointT & origin, const PointT & vector,
    double max_range, double max_angle, double prob_miss, double prob_hit);

  // This function is usually called by insertPointCloud
  // We expose it here to add more control to the user.
  // Once finished adding points, you must call updateFreeCells()
  void addHitPoint(const Vector3D & point, double prob_hit);

  // This function is usually called by insertPointCloud
  // We expose it here to add more control to the user.
  // Once finished adding points, you must call updateFreeCells()
  void addMissPoint(const Vector3D & point, double prob_miss);

  [[nodiscard]] bool isOccupied(const Bonxai::CoordT & coord) const;

  [[nodiscard]] bool isUnknown(const Bonxai::CoordT & coord) const;

  [[nodiscard]] bool isFree(const Bonxai::CoordT & coord) const;

  void getOccupiedVoxels(std::vector<Bonxai::CoordT> & coords);

  void getOccupiedVoxels(std::vector<Bonxai::CoordT> & coords, std::vector<double> & probs);

  void getFreeVoxels(std::vector<Bonxai::CoordT> & coords);

  template<typename PointT>
  void getOccupiedVoxels(std::vector<PointT> & points)
  {
    thread_local std::vector<Bonxai::CoordT> coords;
    coords.clear();
    getOccupiedVoxels(coords);
    for (const auto & coord : coords) {
      const auto p = _grid.coordToPos(coord);
      points.emplace_back(p.x, p.y, p.z);
    }
  }

  template<typename PointT>
  void getOccupiedVoxels(std::vector<PointT> & points, std::vector<double> & probs)
  {
    thread_local std::vector<Bonxai::CoordT> coords;
    coords.clear();
    probs.clear();
    getOccupiedVoxels(coords, probs);
    for (const auto & coord : coords) {
      const auto p = _grid.coordToPos(coord);
      points.emplace_back(p.x, p.y, p.z);
    }
  }

  void updateFreeCells(const Vector3D & origin, double prob_miss);

  void increaseProb(const Vector3D & point, double prob_hit);
  void decreaseProb(const Vector3D & point, double prob_miss);
  void clear();

private:
  VoxelGrid<CellT> _grid;
  int32_t _thres_min_log = logods(0.12f);
  int32_t _thres_max_log = logods(0.97f);
  int32_t _thres_occupancy = logods(0.5);
  uint8_t _update_count = 1;

  std::vector<CoordT> _miss_coords;
  std::vector<CoordT> _hit_coords;

  mutable Bonxai::VoxelGrid<CellT>::Accessor _accessor;
};

//--------------------------------------------------

template<typename PointT, typename Allocator>
inline void ProbabilisticMap::insertPointCloud(
  const std::vector<PointT, Allocator> & points,
  const PointT & position, const PointT & direction,
  double max_range, double max_angle, double prob_miss, double prob_hit)
{
  const auto from = ConvertPoint<Vector3D>(position);
  const auto from_dir = ConvertPoint<Vector3D>(direction);
  const auto from_coord = _grid.posToCoord(from);
  //
  std::vector<CoordT> coords;
  getOccupiedVoxels(coords);
  std::vector<CoordT> ray;
  for (const auto & coord : coords) {
    //
    bool ok = true;
    ray.clear();
    ComputeRay(from_coord, coord, prob_miss, ray);
    for (const auto & ray_coord : ray) {
      CellT * cell = _accessor.value(ray_coord, true);
      if (cell->probability_log > _thres_occupancy) {
        ok = false;
      }
    }
    if (!ok) {continue;}
    //
    const auto to = ConvertPoint<Vector3D>(_grid.coordToPos(coord));
    Vector3D dir = to - from;
    const double range = dir.norm();
    dir.normalize();
    const double angle = std::acos(dir.dot(from_dir));
    if (range < max_range && angle < max_angle) {
      CellT * cell = _accessor.value(coord, true);
      cell->probability_log = std::max(
        cell->probability_log + logods(prob_miss), _thres_min_log);
    }
  }
  //
  for (const auto & point : points) {
    const auto to = ConvertPoint<Vector3D>(point);
    Vector3D dir = to - from;
    const double range = dir.norm();
    dir.normalize();
    if (range >= max_range) {
      addMissPoint(from + dir * max_range, prob_miss);
    } else {
      addHitPoint(to, prob_hit);
    }
  }
  updateFreeCells(from, prob_miss);
}

template<class Functor> inline
void RayIterator(
  const CoordT & key_origin,
  const CoordT & key_end,
  const double prob_miss,
  const Functor & func)
{
  if (key_origin == key_end) {
    return;
  }

  CoordT error = {0, 0, 0};
  CoordT coord = key_origin;
  CoordT delta = (key_end - coord);
  const CoordT step = {delta.x < 0 ? -1 : 1,
    delta.y < 0 ? -1 : 1,
    delta.z < 0 ? -1 : 1};

  delta = {delta.x < 0 ? -delta.x : delta.x,
    delta.y < 0 ? -delta.y : delta.y,
    delta.z < 0 ? -delta.z : delta.z};

  const int max = std::max(std::max(delta.x, delta.y), delta.z);

  // maximum change of any coordinate
  for (int i = 0; i < max - 1; ++i) {
    // update errors
    error = error + delta;
    // manual loop unrolling
    if ((error.x << 1) >= max) {
      coord.x += step.x;
      error.x -= max;
    }
    if ((error.y << 1) >= max) {
      coord.y += step.y;
      error.y -= max;
    }
    if ((error.z << 1) >= max) {
      coord.z += step.z;
      error.z -= max;
    }
    if (!func(coord, prob_miss)) {
      return;
    }
  }
}

}  // namespace Bonxai
