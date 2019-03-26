/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cov_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    cov,
    covOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    covGradient,
    covGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(cov)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
compute cov matrix using LU decomposition
)DOC")
    .Input(
        0,
        "L",
        "2D tensor of Lower triangular matrix"
        "(N, 10).")
    .Output(
        0,
        "covariance matrix",
        "(N, 16)");

OPERATOR_SCHEMA(covGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .Input(
        0,
        "L",
        "See cov.")
    .Input(
        1,
        "d_Y",
        "Gradient")
    .Output(
        0,
        "d_L",
        "Gradient of forward input 0 (L).");

class GetcovGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "covGradient",
        "",
        vector<string>{I(0), GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(cov, GetcovGradient);

} // namespace caffe2
