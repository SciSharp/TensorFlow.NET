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
using Tensorflow.Framework;
using Tensorflow.Framework.Models;

namespace Tensorflow.Keras.Utils
{
    public class tf_utils
    {
        public static bool are_all_symbolic_tensors(Tensor[] tensors)
        {
            return tensors.Select(x => is_symbolic_tensor(x)).Count() == tensors.Length;
        }

        public static bool? constant_value(Tensor pred)
        {
            return smart_module.smart_constant_value(pred);
        }

        public static bool is_symbolic_tensor(Tensor tensor)
        {
            return true;
        }

        public static Tensor[] smart_cond<T>(IVariableV1 pred,
            Func<T[]> true_fn = null,
            Func<T[]> false_fn = null,
            string name = null)
        {
            return control_flow_ops.cond(pred.AsTensor(),
                true_fn: true_fn,
                false_fn: false_fn,
                name: name);
        }

        public static Tensor[] smart_cond<T>(Tensor pred,
            Func<T[]> true_fn = null,
            Func<T[]> false_fn = null,
            string name = null)
        {
            return smart_module.smart_cond(pred,
                true_fn: true_fn,
                false_fn: false_fn,
                name: name);
        }

        public static Tensor smart_cond(bool pred,
            Func<Tensor> true_fn = null,
            Func<Tensor> false_fn = null,
            string name = null)
        {
            return smart_module.smart_cond(pred,
                true_fn: true_fn,
                false_fn: false_fn,
                name: name);
        }

        public static TensorSpec get_tensor_spec(Tensor t, bool dynamic_batch = false, string name = null)
        {
            throw new NotImplementedException("The function is waited to be implemented in the future.");
        }

        public static TensorSpec get_tensor_spec(TensorSpec t, bool dynamic_batch = false, string name = null)
        {
            var spec = t;
            if (!dynamic_batch)
            {
                return spec;
            }
            var dynamic_batch_spec = new TensorSpec(t.shape, t.dtype, t.name);
            var shape = dynamic_batch_spec.shape;
            if(shape.rank > 0)
            {
                var shape_list = shape.as_int_list();
                // TODO(Rinne): check if -1 is equivalent to None in python.
                shape_list[0] = -1;
                dynamic_batch_spec.shape = new Shape(shape_list);
            }
            return dynamic_batch_spec;
        }
    }
}
