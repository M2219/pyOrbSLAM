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

#ifndef G2O_VERTEX_TRACKXYZ_H_
#define G2O_VERTEX_TRACKXYZ_H_

#include <memory>

#include "g2o/config.h"
#include "g2o/core/base_vertex.h"
#include "g2o/core/eigen_types.h"
#include "g2o/core/hyper_graph.h"
#include "g2o/core/hyper_graph_action.h"
#include "g2o/stuff/property.h"
#include "g2o_types_slam3d_api.h"

namespace g2o {
/**
 * \brief Vertex for a tracked point in space
 */
class G2O_TYPES_SLAM3D_API VertexPointXYZ : public BaseVertex<3, Vector3> {
 public:
  VertexPointXYZ() = default;

  void oplusImpl(const VectorX::MapType& update) override {
    estimate_ += update.head<kDimension>();
  }
};

#ifdef G2O_HAVE_OPENGL
/**
 * \brief visualize a 3D point
 */
class VertexPointXYZDrawAction : public DrawAction {
 public:
  VertexPointXYZDrawAction();
  bool operator()(HyperGraph::HyperGraphElement& element,
                  HyperGraphElementAction::Parameters& params_) override;

 protected:
  std::shared_ptr<FloatProperty> pointSize_;
  DrawAction::Parameters* refreshPropertyPtrs(
      HyperGraphElementAction::Parameters& params_) override;
};
#endif

}  // namespace g2o
#endif
