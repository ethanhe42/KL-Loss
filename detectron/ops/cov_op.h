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

#ifndef COV_OP_H_
#define COV_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class covOp final : public Operator<Context> {
 public:
  covOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  int dim_; // dimension for 1 anchor prediction
  Tensor<Context> buff_; // Buffer for element-wise differences
  //Tensor buff_{Context::GetDeviceType()};
};

template <typename T, class Context>
class covGradientOp final : public Operator<Context> {
 public:
  covGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  int dim_; // dimension for 1 anchor prediction
  Tensor<Context> buff_; // Buffer for element-wise differences
  //Tensor buff_{Context::GetDeviceType()};
};

} // namespace caffe2

#endif // COV_OP_H_
