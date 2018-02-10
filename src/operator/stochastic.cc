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
 * \file sotchastic.cc
 * \brief
 * \author Guneet Singh Dhillon
*/

#include "./stochastic-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(StochasticParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new StochasticOp<cpu, DType>(param);
  });
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *StochasticProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                              std::vector<int> *in_type) const {
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(StochasticParam);

MXNET_REGISTER_OP_PROPERTY(Stochastic, StochasticProp)
.describe(R"(Applies stochastic activation pruning operation to input array.
)" ADD_FILELINE)
.add_argument("act", "NDArray-or-Symbol", "activation")
.add_argument("prob", "NDArray-or-Symbol", "probability")
.add_arguments(StochasticParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

