// g2o - General Graph Optimization
// Copyright (C) 2011 R. Kuemmerle, G. Grisetti, W. Burgard
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

#include "odometry_measurement.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <utility>

namespace g2o {

VelocityMeasurement::VelocityMeasurement() : measurement_(0., 0.) {}

VelocityMeasurement::VelocityMeasurement(double vl, double vr, double dt)
    : measurement_(vl, vr), dt_(dt) {}

MotionMeasurement::MotionMeasurement() : measurement_(0., 0., 0.) {}

MotionMeasurement::MotionMeasurement(double x, double y, double theta,
                                     double dt)
    : measurement_(x, y, theta), dt_(dt) {}

MotionMeasurement::MotionMeasurement(Vector3 m, double dt)
    : measurement_(std::move(m)), dt_(dt) {}

VelocityMeasurement OdomConvert::convertToVelocity(const MotionMeasurement& m) {
  if (fabs(m.theta()) > 1e-7) {
    const double translation = std::hypot(m.x(), m.y());
    const double R = translation / (2 * sin(m.theta() / 2));
    double w = 0.;
    if (fabs(m.dt()) > 1e-7) w = m.theta() / m.dt();

    const double vl = (2. * R * w - w) / 2.;
    const double vr = w + vl;

    return VelocityMeasurement(vl, vr, m.dt());
  }
  double vl;
  double vr;
  if (fabs(m.dt()) > 1e-7)
    vl = vr = std::hypot(m.x(), m.y()) / m.dt();
  else
    vl = vr = 0.;
  return VelocityMeasurement(vl, vr, m.dt());
}

MotionMeasurement OdomConvert::convertToMotion(const VelocityMeasurement& v,
                                               double l) {
  double x;
  double y;
  double theta;
  if (fabs(v.vr() - v.vl()) > 1e-7) {
    double R = l * 0.5 * ((v.vl() + v.vr()) / (v.vr() - v.vl()));
    double w = (v.vr() - v.vl()) / l;

    theta = w * v.dt();
    Rotation2D rot(theta);
    Vector2 icc(0, R);
    Vector2 motion = (rot * (Vector2(-1. * icc))) + icc;
    x = motion.x();
    y = motion.y();
  } else {
    double tv = 0.5 * (v.vr() + v.vl());
    theta = 0.;
    x = tv * v.dt();
    y = 0.;
  }

  return MotionMeasurement(x, y, theta, v.dt());
}

}  // namespace g2o