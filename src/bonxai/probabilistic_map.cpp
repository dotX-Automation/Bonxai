#include <bonxai/probabilistic_map.hpp>

namespace Bonxai
{

const int32_t ProbabilisticMap::UnknownProbability = ProbabilisticMap::logods(0.5f);


VoxelGrid<ProbabilisticMap::CellT> & ProbabilisticMap::grid()
{
  return _grid;
}

ProbabilisticMap::ProbabilisticMap(double resolution)
: _grid(resolution), _accessor(_grid.createAccessor())
{}

const VoxelGrid<ProbabilisticMap::CellT> & ProbabilisticMap::grid() const
{
  return _grid;
}

void ProbabilisticMap::setThresMin(const double thres_min)
{
  _thres_min_log = logods(thres_min);
}

double ProbabilisticMap::getThresMin()
{
  return prob(_thres_min_log);
}

void ProbabilisticMap::setThresMax(const double thres_max)
{
  _thres_max_log = logods(thres_max);
}

double ProbabilisticMap::getThresMax()
{
  return prob(_thres_max_log);
}

void ProbabilisticMap::setThresOccupancy(const double thres_occupancy)
{
  _thres_occupancy = logods(thres_occupancy);
}

void ProbabilisticMap::addHitPoint(const Vector3D & point, double prob_hit)
{
  const auto coord = _grid.posToCoord(point);
  CellT * cell = _accessor.value(coord, true);

  if (cell->update_id != _update_count) {
    cell->probability_log = std::min(
      cell->probability_log + logods(prob_hit), _thres_max_log);
    cell->update_id = _update_count;
    _hit_coords.push_back(coord);
  }
}

void ProbabilisticMap::addMissPoint(const Vector3D & point, double prob_miss)
{
  const auto coord = _grid.posToCoord(point);
  CellT * cell = _accessor.value(coord, true);

  if (cell->update_id != _update_count) {
    cell->probability_log = std::max(
      cell->probability_log + logods(prob_miss), _thres_min_log);
    cell->update_id = _update_count;
    _miss_coords.push_back(coord);
  }
}

bool ProbabilisticMap::isOccupied(const CoordT & coord) const
{
  if (auto * cell = _accessor.value(coord, false)) {
    return cell->probability_log > _thres_occupancy;
  }
  return false;
}

bool ProbabilisticMap::isUnknown(const CoordT & coord) const
{
  if (auto * cell = _accessor.value(coord, false)) {
    return cell->probability_log == _thres_occupancy;
  }
  return true;
}

bool ProbabilisticMap::isFree(const CoordT & coord) const
{
  if (auto * cell = _accessor.value(coord, false)) {
    return cell->probability_log < _thres_occupancy;
  }
  return false;
}

void Bonxai::ProbabilisticMap::updateFreeCells(const Vector3D & origin, double prob_miss)
{
  auto accessor = _grid.createAccessor();

  // same as addMissPoint, but using lambda will force inlining
  auto clearPoint = [this, &accessor](const CoordT & coord, double prob_miss)
    {
      CellT * cell = accessor.value(coord, true);
      if (cell->update_id != _update_count) {
        cell->probability_log = std::max(
          cell->probability_log + logods(prob_miss), _thres_min_log);
        cell->update_id = _update_count;
      }
      return true;
    };

  const auto coord_origin = _grid.posToCoord(origin);

  for (const auto & coord_end : _hit_coords) {
    RayIterator(coord_origin, coord_end, prob_miss, clearPoint);
  }
  _hit_coords.clear();

  for (const auto & coord_end : _miss_coords) {
    RayIterator(coord_origin, coord_end, prob_miss, clearPoint);
  }
  _miss_coords.clear();

  if (++_update_count == 4) {
    _update_count = 1;
  }
}

void ProbabilisticMap::getOccupiedVoxels(std::vector<CoordT> & coords)
{
  coords.clear();
  auto visitor = [&](CellT & cell, const CoordT & coord) {
      if (cell.probability_log > _thres_occupancy) {
        coords.push_back(coord);
      }
    };
  _grid.forEachCell(visitor);
}

void ProbabilisticMap::getOccupiedVoxels(std::vector<CoordT> & coords, std::vector<double> & probs)
{
  coords.clear();
  probs.clear();
  auto visitor = [&](CellT & cell, const CoordT & coord) {
      if (cell.probability_log > _thres_occupancy) {
        coords.push_back(coord);
        probs.push_back(prob(cell.probability_log));
      }
    };
  _grid.forEachCell(visitor);
}

void ProbabilisticMap::getFreeVoxels(std::vector<CoordT> & coords)
{
  coords.clear();
  auto visitor = [&](CellT & cell, const CoordT & coord) {
      if (cell.probability_log < _thres_occupancy) {
        coords.push_back(coord);
      }
    };
  _grid.forEachCell(visitor);
}

void ProbabilisticMap::increaseProb(const Vector3D & point, double prob_hit)
{
  const auto coord = _grid.posToCoord(point);
  CellT * cell = _accessor.value(coord, true);
  cell->probability_log = std::min(
    cell->probability_log + logods(prob_hit), _thres_max_log);
}

void ProbabilisticMap::decreaseProb(const Vector3D & point, double prob_miss)
{
  const auto coord = _grid.posToCoord(point);
  CellT * cell = _accessor.value(coord, true);
  cell->probability_log = std::max(
    cell->probability_log + logods(prob_miss), _thres_min_log);
}

void ProbabilisticMap::clear()
{
  auto visitor = [&](CellT & cell, const CoordT & coord) {
      (void) coord;
      cell.probability_log = UnknownProbability;
    };
  _grid.forEachCell(visitor);
}

}  // namespace Bonxai
