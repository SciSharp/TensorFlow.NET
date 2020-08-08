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
using System.Linq;
using Tensorflow.Eager;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class gen_training_ops
    {
        public static Operation resource_apply_adam(Tensor var, Tensor m, Tensor v, Tensor beta1_power, Tensor beta2_power,
            Tensor lr, Tensor beta1, Tensor beta2, Tensor epsilon, Tensor grad,
            bool use_locking = false, bool use_nesterov = false, string name = null)
        {
            if (tf.executing_eagerly())
            {
                var result = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "ResourceApplyAdam", name,
                    null,
                    var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad,
                    "use_locking", use_locking,
                    "use_nesterov", use_nesterov);
                return null;
            }

            throw new NotImplementedException("");
        }

        public static Tensor apply_adam(IVariableV1 var, IVariableV1 m, IVariableV1 v, Tensor beta1_power, Tensor beta2_power, 
            Tensor lr, Tensor beta1, Tensor beta2, Tensor epsilon, Tensor grad, 
            bool use_locking = false, bool use_nesterov = false, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("ApplyAdam", name, new
            {
                var,
                m,
                v,
                beta1_power,
                beta2_power,
                lr,
                beta1,
                beta2,
                epsilon,
                grad,
                use_locking,
                use_nesterov
            });

            return _op.outputs[0];
        }

        public static Tensor apply_gradient_descent(RefVariable var, Tensor alpha, Tensor delta, bool use_locking = false, string name = null)
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

        public static Operation resource_apply_gradient_descent(Tensor var, Tensor alpha, Tensor delta, bool use_locking = false, string name = null)
        {
            if (tf.executing_eagerly())
            {
                var result = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "ResourceApplyGradientDescent", name, 
                    null,
                    var, alpha, delta,
                    "use_locking", use_locking);
                return null;
            }

            var _op = tf.OpDefLib._apply_op_helper("ResourceApplyGradientDescent", name, new
            {
                var,
                alpha,
                delta,
                use_locking
            });

            return _op;
        }
    }
}
