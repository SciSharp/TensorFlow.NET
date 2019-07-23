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

using static Tensorflow.Python;

namespace Tensorflow
{
    public class weights_broadcast_ops
    {
        public static Tensor broadcast_weights(Tensor weights, Tensor values)
        {
            return with(ops.name_scope(null, "broadcast_weights", (weights, values)), scope =>
            {
                values = ops.convert_to_tensor(values, name: "values");
                weights = ops.convert_to_tensor(
                    weights, dtype: values.dtype.as_base_dtype(), name: "weights");

                // Try static check for exact match.
                var weights_shape = weights.TensorShape;
                var values_shape = values.TensorShape;
                if (weights_shape.is_fully_defined() &&
                    values_shape.is_fully_defined())
                    return weights;

                return math_ops.multiply(
                    weights, array_ops.ones_like(values), name: scope);
            });
        }
    }
}
