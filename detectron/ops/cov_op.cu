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

#include "caffe2/core/context_gpu.h"
#include "cov_op.h"

namespace caffe2 {

namespace {
__global__ void kl_selectSmoothL1Kernel(
    const int D, const int H, const int W,
    const int M, const float* Y_hat, const float* Y, const float* L, 
    const float* alpha,
    float* out,
    const float* S, const float beta) {
  // f(x) = 0.5 * x^2 / beta      if |x| < beta
  //        |x| - 0.5 * beta      otherwise
  CUDA_1D_KERNEL_LOOP(i, M) {
    int n = L[i * 4];
    int c = L[i * 4 + 1];
    int y = L[i * 4 + 2];
    int x = L[i * 4 + 3];

    float sq2 = sqrt(2.0);
    for (int j = 0; j < 4; j++){
      // Y_hat: N x (A * CLS * 4) x H x W
      int ind = n * (D * H * W) + (c + j) * (H * W) + y * W + x;
      float y_hat = Y_hat[ind];
      float a = alpha[ind];
      float y = Y[i * 4 + j];
      float val = y_hat - y;
      float abs_val = abs(val);
      if (abs_val < beta) {
        out[ind] = (sq2 * exp(-a) * 0.5 * val * val / beta + 0.5 * sq2 * exp(-a) * beta + a) / max(S[0], 1.0);
      } else {
        out[ind] = (sq2 * exp(-a) * (abs_val - 0.5 * beta) + 0.5 * sq2 * exp(-a) * beta + a) / max(S[0], 1.0);
      }
    }
  }
}


__global__ void kl_selectSmoothL1GradientKernel(
    const int D, const int H, const int W,
    const int M,
    const float* Y_hat,
    const float* Y,
    const float* L,
    const float* alpha,
    float* out,
    float* d_alpha,
    const float* d_loss_data,
    float norm,
    const float* S,
    float beta) {
  // f'(x) = x / beta     if |x| < beta
  //       = sign(x)      otherwise
  // We also scale by norm * d_loss in this kernel for convenience
  CUDA_1D_KERNEL_LOOP(i, M) {
    int n = L[i * 4];
    int c = L[i * 4 + 1];
    int y = L[i * 4 + 2];
    int x = L[i * 4 + 3];
    float d_loss = *d_loss_data;
    float sq2 = sqrt(2.0);

    for (int j = 0; j < 4; j++) {
      int ind = n * (D * H * W) + (c + j) * (H * W) + y * W + x;
      float y_hat = Y_hat[ind];
      float a = alpha[ind];
      float y = Y[i * 4 + j];
      float val = y_hat - y;
      float abs_val = abs(val);
      if (abs_val < beta) {
        out[ind] = norm * d_loss * sq2 * exp(-a) / beta  * val / max(S[0], 1.0);
        d_alpha[ind] = norm * d_loss * (- sq2 * exp(-a) * 0.5 * val * val / beta - 0.5 * sq2 * exp(-a) * beta + 1.0) / max(S[0], 1.0);
      } else {
        out[ind] = norm * d_loss * sq2 * exp(-a) * ((float(0) < val) - (val < float(0))) / max(S[0], 1.0);
        d_alpha[ind] = norm * d_loss * (- sq2* exp(-a) * (abs_val - 0.5 * beta) - 0.5 * sq2 * exp(-a) * beta + 1.0) / max(S[0], 1.0);
      }
    }
  }
}
} // namespace


template<>
bool covOp<float, CUDAContext>::RunOnDevice() {
  // bbox targets predictions, for example: N x (A * 4) H x W in cls-agnostic case
  auto& Y_hat     = Input(0);
  // true targets: for example: M x 4 where M is the #fg boxes per fpn level
  auto& alpha     = Input(1);
  auto& Y         = Input(2);
  // locations of fg boxes: M x 4
  auto& L         = Input(3);
  // total number of fg boxes across all FPN levels: scalar
  auto& S         = Input(4);
  auto* avg_loss  = Output(0);

  avg_loss->Resize(vector<int64_t>());
  if (Y.size() == 0){
    math::Set<float, CUDAContext>(
      1, static_cast<float>(0), avg_loss->mutable_data<float>(), &context_);
    return true;
  }

  int N = Y_hat.dim32(0);
  int D = Y_hat.dim32(1);
  int H = Y_hat.dim32(2);
  int W = Y_hat.dim32(3);

  int M = Y.dim32(0);

  // initialization
  buff_.ResizeLike(Y_hat);
  math::Set<float, CUDAContext>(
    1, static_cast<float>(0), avg_loss->mutable_data<float>(), &context_);
  math::Set<float, CUDAContext>(
    buff_.size(), 0.0, buff_.mutable_data<float>(), &context_);

  // Element-wise smooth l1 loss
  // l := kl_selectSmoothL1((y_hat - y))
  kl_selectSmoothL1Kernel<<<CAFFE_GET_BLOCKS(buff_.size()),
                         CAFFE_CUDA_NUM_THREADS,
                         0, context_.cuda_stream()>>>(
    D, H, W,
    M, Y_hat.data<float>(), Y.data<float>(),
    L.data<float>(), 
    alpha.data<float>(), 
    buff_.mutable_data<float>(),
    S.data<float>(), beta_);

  // Sum of all losses
  // al := sum_i l_i
  float* avg_loss_data = avg_loss->mutable_data<float>();
  math::Sum<float, CUDAContext>(
      buff_.size(), buff_.data<float>(), avg_loss_data, &context_);

  // Average of input batch size
  //math::Scale<float, CUDAContext>(
  math::Scale<float, float, CUDAContext>(
      1, scale_, avg_loss_data, avg_loss_data, &context_);
  return true;
}

template<>
bool covGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& Y_hat      = Input(0);
  auto& alpha      = Input(1);
  auto& Y          = Input(2);
  auto& L          = Input(3);
  auto& S          = Input(4);
  // Below is gradient of net w.r.t. avg_loss ("gradOuput"), should be all 1's
  auto& d_avg_loss = Input(5);
  auto* d_Y_hat    = Output(0); // gradient of net w.r.t. Y_hat ("gradInput")
  auto* d_alpha    = Output(1); // gradient of net w.r.t. alpha ("gradInput")

  d_Y_hat->ResizeLike(Y_hat);
  d_alpha->ResizeLike(alpha);
  math::Set<float, CUDAContext>(
    d_Y_hat->size(), 0.0, d_Y_hat->mutable_data<float>(), &context_);
  math::Set<float, CUDAContext>(
    d_alpha->size(), 0.0, d_alpha->mutable_data<float>(), &context_);
  if (Y.size() == 0){
    return true;
  }

  int N = Y_hat.dim32(0);
  int D = Y_hat.dim32(1);
  int H = Y_hat.dim32(2);
  int W = Y_hat.dim32(3);

  int M = Y.dim32(0);
  // Element-wise weighted difference (can be used to ignore or reweight
  // specific components)
  // d := (y_hat - y)
  // d_Y_hat := d_avg_loss * kl_selectSmoothL1'((y_hat - y))

  kl_selectSmoothL1GradientKernel<<<CAFFE_GET_BLOCKS(d_Y_hat->size()),
                                 CAFFE_CUDA_NUM_THREADS,
                                 0, context_.cuda_stream()>>>(
    D, H, W, M, Y_hat.data<float>(), Y.data<float>(),
    L.data<float>(), 
    alpha.data<float>(),
    d_Y_hat->mutable_data<float>(),
    d_alpha->mutable_data<float>(),
    d_avg_loss.data<float>(), scale_, S.data<float>(), beta_);

  return true;
}


REGISTER_CUDA_OPERATOR(cov,
                       covOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(covGradient,
                       covGradientOp<float, CUDAContext>);
}  // namespace caffe2
