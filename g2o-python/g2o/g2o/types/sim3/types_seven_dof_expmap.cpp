// g2o - General Graph Optimization
// Copyright (C) 2011 H. Strasdat
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

#include "types_seven_dof_expmap.h"

#include <Eigen/Core>

#include "g2o/core/factory.h"
#include "g2o/types/sim3/sim3.h"
#include "g2o/types/slam3d/se3_ops.h"

namespace g2o {

G2O_USE_TYPE_GROUP(sba);
G2O_REGISTER_TYPE_GROUP(sim3);

G2O_REGISTER_TYPE_NAME("VERTEX_SIM3:EXPMAP", VertexSim3Expmap);
G2O_REGISTER_TYPE_NAME("EDGE_SIM3:EXPMAP", EdgeSim3);
G2O_REGISTER_TYPE_NAME("EDGE_PROJECT_SIM3_XYZ:EXPMAP", EdgeSim3ProjectXYZ);
G2O_REGISTER_TYPE_NAME("EDGE_PROJECT_INVERSE_SIM3_XYZ:EXPMAP",
                       EdgeInverseSim3ProjectXYZ);

VertexSim3Expmap::VertexSim3Expmap() {
  marginalized_ = false;
  _fix_scale = false;

  _principle_point1[0] = 0;
  _principle_point1[1] = 0;
  _focal_length1[0] = 1;
  _focal_length1[1] = 1;

  _principle_point2[0] = 0;
  _principle_point2[1] = 0;
  _focal_length2[0] = 1;
  _focal_length2[1] = 1;
}

void VertexSim3Expmap::oplusImpl(const VectorX::MapType& update) {
  if (_fix_scale) {
    auto& update_non_const = const_cast<VectorX::MapType&>(update);
    update_non_const[6] = 0;
  }

  Sim3 s(update);
  setEstimate(s * estimate());
}

Vector2 VertexSim3Expmap::cam_map1(const Vector2& v) const {
  Vector2 res;
  res[0] = v[0] * _focal_length1[0] + _principle_point1[0];
  res[1] = v[1] * _focal_length1[1] + _principle_point1[1];
  return res;
}

Vector2 VertexSim3Expmap::cam_map2(const Vector2& v) const {
  Vector2 res;
  res[0] = v[0] * _focal_length2[0] + _principle_point2[0];
  res[1] = v[1] * _focal_length2[1] + _principle_point2[1];
  return res;
}

void EdgeSim3::computeError() {
  const VertexSim3Expmap* v1 = vertexXnRaw<0>();
  const VertexSim3Expmap* v2 = vertexXnRaw<1>();

  Sim3 C(measurement_);
  Sim3 err = C * v1->estimate() * v2->estimate().inverse();
  error_ = err.log();
}

double EdgeSim3::initialEstimatePossible(const OptimizableGraph::VertexSet&,
                                         OptimizableGraph::Vertex*) {
  return 1.;
}
void EdgeSim3::initialEstimate(const OptimizableGraph::VertexSet& from,
                               OptimizableGraph::Vertex* /*to*/) {
  auto v1 = vertexXn<0>();
  auto v2 = vertexXn<1>();
  if (from.count(v1) > 0)
    v2->setEstimate(measurement() * v1->estimate());
  else
    v1->setEstimate(measurement().inverse() * v2->estimate());
}

#if G2O_SIM3_JACOBIAN
void EdgeSim3::linearizeOplus() {
  VertexSim3Expmap* v1 = vertexXnRaw<0>();
  VertexSim3Expmap* v2 = vertexXnRaw<1>();
  const Sim3 Si(v1->estimate());  // Siw
  const Sim3 Sj(v2->estimate());

  const Sim3& Sji = measurement_;

  // error in Lie Algebra
  const Eigen::Matrix<double, 7, 1> error = (Sji * Si * Sj.inverse()).log();
  const Eigen::Vector3d phi = error.block<3, 1>(0, 0);  // rotation
  const Eigen::Vector3d tau = error.block<3, 1>(3, 0);  // translation
  const double s = error(6);                            // scale

  const Eigen::Matrix<double, 7, 7> I7 =
      Eigen::Matrix<double, 7, 7>::Identity();
  const Eigen::Matrix<double, 3, 3> I3 =
      Eigen::Matrix<double, 3, 3>::Identity();

  // Jacobi Matrix of Si
  // note: because the order of rotation and translation is different,
  //       so it is slightly different from the formula.
  Eigen::Matrix<double, 7, 7> jacobi_i = Eigen::Matrix<double, 7, 7>::Zero();
  jacobi_i.block<3, 3>(0, 0) = -skew(phi);
  jacobi_i.block<3, 3>(3, 3) = -(skew(phi) + s * I3);
  jacobi_i.block<3, 3>(3, 0) = -skew(tau);
  jacobi_i.block<3, 1>(3, 6) = tau;

  // Adjoint matrix of Sji
  Eigen::Matrix<double, 7, 7> adj_Sji = I7;
  adj_Sji.block<3, 3>(0, 0) = Sji.rotation().toRotationMatrix();
  adj_Sji.block<3, 3>(3, 3) = Sji.scale() * Sji.rotation().toRotationMatrix();
  adj_Sji.block<3, 3>(3, 0) =
      skew(Sji.translation()) * Sji.rotation().toRotationMatrix();
  adj_Sji.block<3, 1>(3, 6) = -Sji.translation();

  _jacobianOplusXi = (I7 + 0.5 * jacobi_i) * adj_Sji;

  // Jacobi Matrix of Sj
  Eigen::Matrix<double, 7, 7> jacobi_j = Eigen::Matrix<double, 7, 7>::Zero();
  jacobi_j.block<3, 3>(0, 0) = skew(phi);
  jacobi_j.block<3, 3>(3, 3) = skew(phi) + s * I3;
  jacobi_j.block<3, 3>(3, 0) = skew(tau);
  jacobi_j.block<3, 1>(3, 6) = -tau;

  _jacobianOplusXj = -(I7 + 0.5 * jacobi_j);
}
#endif

/**Sim3ProjectXYZ*/

void EdgeSim3ProjectXYZ::computeError() {
  const VertexSim3Expmap* v1 = vertexXnRaw<1>();
  const VertexPointXYZ* v2 = vertexXnRaw<0>();

  Vector2 obs(measurement_);
  error_ = obs - v1->cam_map1(project(v1->estimate().map(v2->estimate())));
}

void EdgeInverseSim3ProjectXYZ::computeError() {
  const VertexSim3Expmap* v1 = vertexXnRaw<1>();
  const VertexPointXYZ* v2 = vertexXnRaw<0>();

  Vector2 obs(measurement_);
  error_ =
      obs - v1->cam_map2(project(v1->estimate().inverse().map(v2->estimate())));
}

//  void EdgeSim3ProjectXYZ::linearizeOplus()
//  {
//    VertexSim3Expmap * vj = static_cast<VertexSim3Expmap *>(_vertices[1]);
//    Sim3 T = vj->estimate();

//    VertexPointXYZ* vi = static_cast<VertexPointXYZ*>(_vertices[0]);
//    Vector3 xyz = vi->estimate();
//    Vector3 xyz_trans = T.map(xyz);

//    double x = xyz_trans[0];
//    double y = xyz_trans[1];
//    double z = xyz_trans[2];
//    double z_2 = z*z;

//    Matrix<double,2,3,Eigen::ColMajor> tmp;
//    tmp(0,0) = _focal_length(0);
//    tmp(0,1) = 0;
//    tmp(0,2) = -x/z*_focal_length(0);

//    tmp(1,0) = 0;
//    tmp(1,1) = _focal_length(1);
//    tmp(1,2) = -y/z*_focal_length(1);

//    _jacobianOplusXi =  -1./z * tmp * T.rotation().toRotationMatrix();

//    _jacobianOplusXj(0,0) =  x*y/z_2 * _focal_length(0);
//    _jacobianOplusXj(0,1) = -(1+(x*x/z_2)) *_focal_length(0);
//    _jacobianOplusXj(0,2) = y/z *_focal_length(0);
//    _jacobianOplusXj(0,3) = -1./z *_focal_length(0);
//    _jacobianOplusXj(0,4) = 0;
//    _jacobianOplusXj(0,5) = x/z_2 *_focal_length(0);
//    _jacobianOplusXj(0,6) = 0; // scale is ignored

//    _jacobianOplusXj(1,0) = (1+y*y/z_2) *_focal_length(1);
//    _jacobianOplusXj(1,1) = -x*y/z_2 *_focal_length(1);
//    _jacobianOplusXj(1,2) = -x/z *_focal_length(1);
//    _jacobianOplusXj(1,3) = 0;
//    _jacobianOplusXj(1,4) = -1./z *_focal_length(1);
//    _jacobianOplusXj(1,5) = y/z_2 *_focal_length(1);
//    _jacobianOplusXj(1,6) = 0; // scale is ignored
//  }

}  // namespace g2o
