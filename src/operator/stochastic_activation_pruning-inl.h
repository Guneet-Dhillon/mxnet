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
 * \file stochastic_activation_pruning-inl.h
 * \brief
 * \author Guneet Singh Dhillon
*/

#ifndef MXNET_OPERATOR_STOCHASTIC_INL_H_
#define MXNET_OPERATOR_STOCHASTIC_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <./math.h>
#include <random>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include "./operator_common.h"
#include "./mshadow_op.h"

#if defined(USE_MKL) && defined(_OPENMP)
#include <omp.h>

#include <mkl_vml_functions.h>
#include <mkl_vsl.h>
#endif  // USE_MKL && _OPENMP


namespace stochastic_activation_pruning {
enum StochasticActivationPruningOpInputs {kAct, kProb};
enum StochasticActivationPruningOpOutputs {kOut, kMask};
}  // namespace stochastic_activation_pruning

namespace mxnet {
namespace op {

struct StochasticActivationPruningParam :
  public dmlc::Parameter<StochasticActivationPruningParam> {
  float frac;
  DMLC_DECLARE_PARAMETER(StochasticActivationPruningParam) {
    DMLC_DECLARE_FIELD(frac).set_default(1.0)
    .describe("Fraction of the input that need to be sampled.");
  }
};  // struct StochasticActivationPruningParam

template<typename xpu, typename DType>
class StochasticActivationPruningOp : public Operator {
 public:
  explicit StochasticActivationPruningOp(
    StochasticActivationPruningParam param) {
    this->frac_ = param.frac;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 2U);
    CHECK_EQ(out_data.size(), 2U);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> act =
      in_data[stochastic_activation_pruning::kAct].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> prob =
      in_data[stochastic_activation_pruning::kProb].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> mask =
      out_data[stochastic_activation_pruning::kMask].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> out =
      out_data[stochastic_activation_pruning::kOut].FlatTo2D<xpu, DType>(s);

#if !defined(__CUDACC__)
    DType* act_ptr = act.dptr_;
    DType* prob_ptr = prob.dptr_;
    DType* mask_ptr = mask.dptr_;
    DType* out_ptr = out.dptr_;

    const int mask_row = mask.shape_[0];
    const int mask_col = mask.shape_[1];
    const int mask_size = mask_row * mask_col;

    const int k = int(frac_ * mask_col);

    // Initialize mask
  #pragma omp parallel for
    for (int i = 0; i < mask_size; ++i) {
      mask_ptr[i] = 0.0f;
    }

    // Sample activations
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen (seed);
  #pragma omp parallel for
    for (int i = 0; i < mask_row; ++i) {
      std::discrete_distribution<int> dist (&(prob_ptr[i * mask_col]),
        &(prob_ptr[(i+1) * mask_col]));
    #pragma omp parallel for
      for (int j = 0; j < k; ++j) {
        int index = (i * mask_col) + dist(gen);
        mask_ptr[index] =
          DType(1.0 / (1.0 - pow(1.0 - double(prob_ptr[index]), k)));
      }
    }

    // Output
  #pragma omp parallel for
    for (int i = 0; i < mask_size; ++i) {
      out_ptr[i] = act_ptr[i] * mask_ptr[i];
    }

#else
    const int k = int(frac_ * mask.shape_[1]);
    forward_GPU(act, prob, mask, out, k);

#endif
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1U);
    CHECK_EQ(in_grad.size(), 2U);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> grad =
      out_grad[stochastic_activation_pruning::kOut].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> mask =
      out_data[stochastic_activation_pruning::kMask].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> act =
      in_grad[stochastic_activation_pruning::kAct].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> prob =
      in_grad[stochastic_activation_pruning::kProb].FlatTo2D<xpu, DType>(s);

#if !defined(__CUDACC__)
    DType* grad_ptr = grad.dptr_;
    DType* mask_ptr = mask.dptr_;
    DType* act_ptr = act.dptr_;
    DType* prob_ptr = prob.dptr_;

    const int mask_size = mask.shape_[0] * mask.shape_[1];

  #pragma omp parallel for
    for (int i = 0; i < mask_size; ++i) {
      act_ptr[i] = grad_ptr[i] * mask_ptr[i];
      prob_ptr[i] = 0.0f;
    }

#else
    backward_GPU(grad, mask, act, prob);

#endif
  }

 private:
  real_t frac_;
};  // class StochasticActivationPruningOp

template<typename xpu>
Operator *CreateOp(StochasticActivationPruningParam param, int dtype);

#if DMLC_USE_CXX11
class StochasticActivationPruningProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs)
    override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 2U);
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;
    out_shape->clear();
    out_shape->push_back(dshape);
    out_shape->push_back(dshape);
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_EQ(in_type->size(), 2U);
    int dtype = in_type->at(0);

    if (dtype == -1) {
      LOG(FATAL) << "input type is not specified.";
      return false;
    }

    size_t nout = this->ListOutputs().size();
    out_type->clear();
    for (size_t i = 0; i < nout; ++i) out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new StochasticActivationPruningProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "StochasticActivationPruning";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[stochastic_activation_pruning::kOut],
      out_data[stochastic_activation_pruning::kMask]};
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  int NumOutputs() const override {
    return 2;
  }

  std::vector<std::string> ListArguments() const override {
    return {"act", "prob"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "mask"};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "not Implemented";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  StochasticActivationPruningParam param_;
};  // class StochasticActivationPruningProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_STOCHASTIC_INL_H_
