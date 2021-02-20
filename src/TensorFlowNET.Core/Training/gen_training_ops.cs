/*****************************************************************************
   Copyright 2018 The TensorFlow.NET Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

using System;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class gen_training_ops
    {
        public static Tensor resource_apply_adam(Tensor var, Tensor m, Tensor v, Tensor beta1_power, Tensor beta2_power,
            Tensor lr, Tensor beta1, Tensor beta2, Tensor epsilon, Tensor grad,
            bool use_locking = false, bool use_nesterov = false, string name = null)
                => tf.Context.ExecuteOp("ResourceApplyAdam", name, 
                    new ExecuteOpArgs(var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad)
                        .SetAttributes(new { use_locking, use_nesterov }));

        public static Tensor apply_adam(Tensor var, Tensor m, Tensor v, Tensor beta1_power, Tensor beta2_power,
            Tensor lr, Tensor beta1, Tensor beta2, Tensor epsilon, Tensor grad,
            bool use_locking = false, bool use_nesterov = false, string name = null)
                => tf.Context.ExecuteOp("ApplyAdam", name, 
                    new ExecuteOpArgs(var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad)
                        .SetAttributes(new { use_locking, use_nesterov }));

        public static Tensor apply_gradient_descent(IVariableV1 var, Tensor alpha, Tensor delta, bool use_locking = false, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("ApplyGradientDescent", name, new
            {
                var,
                alpha,
                delta,
                use_locking
            });

            return _op.output;
        }

        public static Tensor resource_apply_gradient_descent(Tensor var, Tensor alpha, Tensor delta, bool use_locking = false, string name = null)
            => tf.Context.ExecuteOp("ResourceApplyGradientDescent", name, 
                new ExecuteOpArgs(var, alpha, delta).SetAttributes(new { use_locking }));
    }
}
