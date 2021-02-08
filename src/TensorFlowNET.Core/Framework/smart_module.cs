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
using static Tensorflow.Binding;

namespace Tensorflow.Framework
{
    public class smart_module
    {
        public static Tensor[] smart_cond<T>(Tensor pred,
            Func<T[]> true_fn = null,
            Func<T[]> false_fn = null,
            string name = null)
        {
            var pred_value = smart_constant_value(pred);
            if (pred_value.HasValue)
            {
                if (pred_value.Value)
                    return true_fn() as Tensor[];
                else
                    return false_fn() as Tensor[];
            }
            else
                return control_flow_ops.cond(pred,
                    true_fn: true_fn,
                    false_fn: false_fn,
                    name: name);
        }

        public static Tensor smart_cond(bool pred,
            Func<Tensor> true_fn = null,
            Func<Tensor> false_fn = null,
            string name = null)
        {
            return pred ? true_fn() : false_fn();
        }

        public static bool? smart_constant_value(Tensor pred)
        {
            var pred_value = tensor_util.constant_value(pred);
            if (pred_value is null)
            {
                var result = range(pred.op.NumOutputs).Select(x => IntPtr.Zero).ToArray();
                var evaluated = c_api.TF_TryEvaluateConstant(pred.graph, pred._as_tf_output(), result, tf.Status.Handle);
                if (!evaluated || c_api.TF_GetCode(tf.Status.Handle) != TF_Code.TF_OK)
                    return null;
                else
                    throw new NotImplementedException("");
            }

            return pred_value;
        }
    }
}
