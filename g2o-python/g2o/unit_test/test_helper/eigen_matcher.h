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

#include <gmock/gmock.h>

#include <Eigen/Core>

namespace g2o::internal {

MATCHER_P(EigenEqual, expect,
          std::string(negation ? "isn't" : "is") + " equal to" +
              ::testing::PrintToString(expect)) {
  return arg == expect;
}

MATCHER_P2(EigenApproxEqual, expect, prec,
           std::string(negation ? "isn't" : "is") + " approx equal to" +
               ::testing::PrintToString(expect) + "\nwith precision " +
               ::testing::PrintToString(prec)) {
  return arg.isApprox(expect, prec);
}

MATCHER_P2(EigenMaxDiff, expect, max_diff,
           std::string(negation ? "doesn't" : "does") + " differ from " +
               ::testing::PrintToString(expect) + "\nmore than " +
               ::testing::PrintToString(max_diff)) {
  return (arg - expect).array().abs().maxCoeff() <= max_diff;
}

template <class Base>
class EigenPrintWrap : public Base {
  void PrintTo(const EigenPrintWrap& m, ::std::ostream* o) { *o << "\n" << m; }
};

template <class Base>
const EigenPrintWrap<Base>& print_wrap(const Base& base) {
  return static_cast<const EigenPrintWrap<Base>&>(base);
}

}  // namespace g2o::internal
