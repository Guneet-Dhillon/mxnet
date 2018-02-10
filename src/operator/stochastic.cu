/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file stochastic.cc
 * \brief
 * \author Guneet Singh Dhillon
*/

#include "./stochastic-inl.h"

#include <random>
#include <./math.h>

namespace mshadow {
namespace cuda {

template<typename DType>
__global__ void forward_kernel_0(DType *mask_ptr, int mask_size) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index < mask_size)
    mask_ptr[index] = 0.0f;
}

template<typename DType>
__global__ void forward_kernel_1(DType *out_ptr, DType *act_ptr, DType *mask_ptr,
  int mask_size) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index < mask_size)
      out_ptr[index] = act_ptr[index] * mask_ptr[index];
}

template<typename DType>
__global__ void backward_kernel(DType *act_ptr, DType *grad_ptr, DType *mask_ptr,
  DType *prob_ptr, int mask_size) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index < mask_size) {
      act_ptr[index] = grad_ptr[index] * mask_ptr[index];
      prob_ptr[index] = 0.0f;
    }
}

} // namespace cuda

template<typename xpu, typename DType>
inline void forward_GPU(
  Tensor<xpu, 2, DType> act,
  Tensor<xpu, 2, DType> prob,
  Tensor<xpu, 2, DType> mask,
  Tensor<xpu, 2, DType> out,
  int k) {

    DType* act_ptr = act.dptr_;
    DType* prob_ptr = prob.dptr_;
    DType* mask_ptr = mask.dptr_;
    DType* out_ptr = out.dptr_;

    const int mask_row = mask.shape_[0];
    const int mask_col = mask.shape_[1];
    const int mask_size = mask_row * mask_col;

    cuda::forward_kernel_0<DType><<<(mask_size / 512) + 1, 512>>>(mask_ptr, mask_size);

    int size = mask_size * sizeof(DType);
    DType* host_mask_ptr = (DType*) malloc(size);
    DType* host_prob_ptr = (DType*) malloc(size);
    cudaMemcpy(host_mask_ptr, mask_ptr, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_prob_ptr, prob_ptr, size, cudaMemcpyDeviceToHost);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen (seed);
  #pragma omp parallel for
    for (int i = 0; i < mask_row; ++i) {
      std::discrete_distribution<int> dist (&(host_prob_ptr[i * mask_col]), &(host_prob_ptr[(i+1) * mask_col]));
    #pragma omp parallel for
      for (int j = 0; j < k; ++j) {
        int index = (i * mask_col) + dist(gen);
        host_mask_ptr[index] = DType(1.0 / (1.0 - pow(1.0 - double(host_prob_ptr[index]), k)));
      }
    }
    cudaMemcpy(mask_ptr, host_mask_ptr, size, cudaMemcpyHostToDevice);
    free(host_mask_ptr);
    free(host_prob_ptr);

    cuda::forward_kernel_1<DType><<<(mask_size / 512) + 1, 512>>>(out_ptr, act_ptr,
      mask_ptr, mask_size);
}

template<typename xpu, typename DType>
inline void backward_GPU(
  Tensor<xpu, 2, DType> grad,
  Tensor<xpu, 2, DType> mask,
  Tensor<xpu, 2, DType> act,
  Tensor<xpu, 2, DType> prob) {

    DType* grad_ptr = grad.dptr_;
    DType* mask_ptr = mask.dptr_;
    DType* act_ptr = act.dptr_;
    DType* prob_ptr = prob.dptr_;

    const int mask_size = mask.shape_[0] * mask.shape_[1];

    cuda::backward_kernel<DType><<<(mask_size / 512) + 1, 512>>>(
      act_ptr, grad_ptr, mask_ptr, prob_ptr, mask_size);
}

} // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<gpu>(StochasticParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new StochasticOp<gpu, DType>(param);
  });
  return op;
}
}  // namespace op
}  // namespace mxnet

