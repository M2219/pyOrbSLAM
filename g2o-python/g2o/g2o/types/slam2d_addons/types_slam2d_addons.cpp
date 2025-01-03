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

#include "g2o/config.h"
#include "g2o/core/factory.h"
#include "g2o/core/hyper_graph_action.h"
#include "g2o/types/slam2d_addons/edge_line2d.h"
#include "g2o/types/slam2d_addons/edge_line2d_pointxy.h"
#include "g2o/types/slam2d_addons/edge_se2_line2d.h"
#include "g2o/types/slam2d_addons/edge_se2_segment2d.h"
#include "g2o/types/slam2d_addons/edge_se2_segment2d_line.h"
#include "g2o/types/slam2d_addons/edge_se2_segment2d_pointLine.h"
#include "g2o/types/slam2d_addons/vertex_line2d.h"
#include "g2o/types/slam2d_addons/vertex_segment2d.h"

namespace g2o {

G2O_REGISTER_TYPE_GROUP(slam2d_segment);

G2O_REGISTER_TYPE_NAME("VERTEX_SEGMENT2D", VertexSegment2D);
G2O_REGISTER_TYPE_NAME("VERTEX_LINE2D", VertexLine2D);

G2O_REGISTER_TYPE_NAME("EDGE_SE2_SEGMENT2D", EdgeSE2Segment2D);
G2O_REGISTER_TYPE_NAME("EDGE_SE2_SEGMENT2D_LINE", EdgeSE2Segment2DLine);
G2O_REGISTER_TYPE_NAME("EDGE_SE2_SEGMENT2D_POINTLINE",
                       EdgeSE2Segment2DPointLine);
G2O_REGISTER_TYPE_NAME("EDGE_SE2_LINE2D", EdgeSE2Line2D);
G2O_REGISTER_TYPE_NAME("EDGE_LINE2D", EdgeLine2D);
G2O_REGISTER_TYPE_NAME("EDGE_LINE2D_POINTXY", EdgeLine2DPointXY);

#ifdef G2O_HAVE_OPENGL
G2O_REGISTER_ACTION(VertexSegment2DDrawAction);
#endif

}  // namespace g2o
