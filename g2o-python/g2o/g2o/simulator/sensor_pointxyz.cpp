// g2o - General Graph Optimization
// Copyright (C) 2011 G. Grisetti, R. Kuemmerle, W. Burgard
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "sensor_pointxyz.h"

#include <cassert>
#include <utility>

#include "g2o/core/eigen_types.h"
#include "g2o/simulator/simulator.h"

namespace g2o {

// SensorPointXYZ
SensorPointXYZ::SensorPointXYZ(std::string name)
    : BinarySensor<Robot3D, EdgeSE3PointXYZ, WorldObjectTrackXYZ>(
          std::move(name)) {
  setInformation(Vector3(1000., 1000., 10.).asDiagonal());
}

bool SensorPointXYZ::isVisible(SensorPointXYZ::WorldObjectType* to) {
  if (!robotPoseVertex_) return false;
  assert(to && to->vertex());
  const VertexType::EstimateType& pose = to->vertex()->estimate();
  const VertexType::EstimateType delta = sensorPose_.inverse() * pose;
  const double range2 = delta.squaredNorm();
  if (range2 > maxRange2_) return false;
  if (range2 < minRange2_) return false;
  // the cameras have the z in front
  double bearing = acos(delta.normalized().z());
  return fabs(bearing) <= fov_;
}

void SensorPointXYZ::addParameters(World& world) {
  if (!offsetParam_) offsetParam_ = std::make_shared<ParameterSE3Offset>();
  world.addParameter(offsetParam_);
}

void SensorPointXYZ::addNoise(EdgeType* e) {
  EdgeType::ErrorVector n = sampler_.generateSample();
  e->setMeasurement(e->measurement() + n);
  e->setInformation(information());
}

void SensorPointXYZ::sense(BaseRobot& robot, World& world) {
  if (!offsetParam_) {
    return;
  }
  robotPoseVertex_ = robotPoseVertex<PoseVertexType>(robot, world);
  if (!robotPoseVertex_) return;
  sensorPose_ = robotPoseVertex_->estimate() * offsetParam_->param();
  for (const auto& it : world.objects()) {
    auto* o = dynamic_cast<WorldObjectType*>(it.get());
    if (!o || !isVisible(o)) continue;
    auto e = mkEdge(o);
    if (!e) continue;
    e->setParameterId(0, offsetParam_->id());
    world.graph().addEdge(e);
    e->setMeasurementFromState();
    addNoise(e.get());
  }
}

}  // namespace g2o
