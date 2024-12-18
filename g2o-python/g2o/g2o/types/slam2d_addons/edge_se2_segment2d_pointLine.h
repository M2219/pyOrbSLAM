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

#ifndef G2O_EDGE_SE2_SEGMENT2D_POINTLINE_H
#define G2O_EDGE_SE2_SEGMENT2D_POINTLINE_H

#include <Eigen/Core>

#include "g2o/core/base_binary_edge.h"
#include "g2o/core/eigen_types.h"
#include "g2o/types/slam2d/vertex_se2.h"
#include "g2o_types_slam2d_addons_api.h"
#include "vertex_segment2d.h"

namespace g2o {

class G2O_TYPES_SLAM2D_ADDONS_API EdgeSE2Segment2DPointLine
    : public BaseBinaryEdge<3, Vector3, VertexSE2, VertexSegment2D> {
 public:
  [[nodiscard]] double theta() const { return measurement_[2]; }
  [[nodiscard]] Vector2 point() const {
    Eigen::Map<const Vector2> p(measurement_.data());
    return p;
  }

  void setTheta(double t) { measurement_[2] = t; }
  void setPoint(const Vector2& p_) {
    Eigen::Map<Vector2> p(measurement_.data());
    p = p_;
  }

  [[nodiscard]] int pointNum() const { return pointNum_; }
  void setPointNum(int pn) { pointNum_ = pn; }

  void computeError() override;

  bool setMeasurementFromState() override;

 protected:
  int pointNum_ = 0;

  /* #ifndef NUMERIC_JACOBIAN_TWO_D_TYPES */
  /*       virtual void linearizeOplus(); */
  /* #endif */
};

}  // namespace g2o

#endif
