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

#ifndef G2O_EDGE_SE2_POINT_XY_H
#define G2O_EDGE_SE2_POINT_XY_H

#include <Eigen/Core>

#include "g2o/config.h"
#include "g2o/core/base_binary_edge.h"
#include "g2o/core/eigen_types.h"
#include "g2o/core/hyper_graph.h"
#include "g2o/core/hyper_graph_action.h"
#include "g2o/core/optimizable_graph.h"
#include "g2o_types_slam2d_api.h"
#include "se2.h"
#include "vertex_point_xy.h"
#include "vertex_se2.h"

namespace g2o {

class G2O_TYPES_SLAM2D_API EdgeSE2PointXY
    : public BaseBinaryEdge<2, Vector2, VertexSE2, VertexPointXY> {
 public:
  EdgeSE2PointXY() = default;

  void computeError() override {
    const VertexSE2* v1 = vertexXnRaw<0>();
    const VertexPointXY* l2 = vertexXnRaw<1>();
    error_ = (v1->estimate().inverse() * l2->estimate()) - measurement_;
  }

  bool setMeasurementFromState() override {
    const VertexSE2* v1 = vertexXnRaw<0>();
    const VertexPointXY* l2 = vertexXnRaw<1>();
    measurement_ = v1->estimate().inverse() * l2->estimate();
    return true;
  }

  void initialEstimate(const OptimizableGraph::VertexSet& from,
                       OptimizableGraph::Vertex* to) override;
  double initialEstimatePossible(const OptimizableGraph::VertexSet& from,
                                 OptimizableGraph::Vertex* to) override {
    (void)to;
    return (from.count(vertices_[0]) == 1 ? 1.0 : -1.0);
  }
#ifndef NUMERIC_JACOBIAN_TWO_D_TYPES
  void linearizeOplus() override;
#endif
};

#ifdef G2O_HAVE_OPENGL
class G2O_TYPES_SLAM2D_API EdgeSE2PointXYDrawAction : public DrawAction {
 public:
  EdgeSE2PointXYDrawAction();
  bool operator()(HyperGraph::HyperGraphElement& element,
                  HyperGraphElementAction::Parameters& params_) override;
};
#endif

}  // namespace g2o

#endif