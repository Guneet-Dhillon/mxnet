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
 * \file stochastic_activation_pruning.cc
 * \brief
 * \author Guneet Singh Dhillon
*/

#include "./stochastic_activation_pruning-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(StochasticActivationPruningParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new StochasticActivationPruningOp<cpu, DType>(param);
  });
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *StochasticActivationPruningProp::CreateOperatorEx(Context ctx,
  std::vector<TShape> *in_shape, std::vector<int> *in_type) const {
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(StochasticActivationPruningParam);

MXNET_REGISTER_OP_PROPERTY(StochasticActivationPruning,
  StochasticActivationPruningProp)
.describe(R"(Applies stochastic activation pruning on activation map input.
)" ADD_FILELINE)
.add_argument("act", "NDArray-or-Symbol", "flattened activation map")
.add_argument("prob", "NDArray-or-Symbol",
  "probabilities corresponding to the activations")
.add_arguments(StochasticActivationPruningParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
