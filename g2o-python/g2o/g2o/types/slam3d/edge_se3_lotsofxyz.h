// g2o - General Graph Optimization
// Copyright (C) 2011 R. Kuemmerle, G. Grisetti, H. Strasdat, W. Burgard
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

#ifndef G2O_SE3_LOTSOF_XYZ
#define G2O_SE3_LOTSOF_XYZ

#include "g2o/core/base_variable_sized_edge.h"
#include "g2o/core/eigen_types.h"
#include "g2o/core/optimizable_graph.h"
#include "g2o_types_slam3d_api.h"

namespace g2o {

class G2O_TYPES_SLAM3D_API EdgeSE3LotsOfXYZ
    : public BaseVariableSizedEdge<-1, VectorX> {
 public:
  EdgeSE3LotsOfXYZ();

  void resize(size_t size) override {
    BaseVariableSizedEdge<-1, VectorX>::resize(size);
    int observed_points = size > 0 ? size - 1 : 0;
    measurement_.resize(observed_points * 3L, 1);
    setDimension(observed_points * 3);
  }

  void computeError() override;

  bool setMeasurementFromState() override;

  void initialEstimate(const OptimizableGraph::VertexSet&,
                       OptimizableGraph::Vertex*) override;
  double initialEstimatePossible(const OptimizableGraph::VertexSet&,
                                 OptimizableGraph::Vertex*) override;

  void linearizeOplus() override;
};

}  // namespace g2o

#endif  // G2O_SE3_LOTSOF_XYZ
