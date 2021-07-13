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

namespace Tensorflow.Operations
{
    public class rnn_cell_impl
    {
        public BasicRnnCell BasicRNNCell(int num_units)
            => new BasicRnnCell(num_units);

        public static Tensor _concat(Tensor prefix, int suffix, bool @static = false)
        {
            var p = prefix;
            var p_static = tensor_util.constant_value(prefix);
            if (p.ndim == 0)
                p = array_ops.expand_dims(p, 0);
            else if (p.ndim != 1)
                throw new ValueError($"prefix tensor must be either a scalar or vector, but saw tensor: {p}");

            var s_tensor_shape = new Shape(suffix);
            var s_static = s_tensor_shape.ndim > -1 ?
                s_tensor_shape.dims :
                null;
            var s = s_tensor_shape.IsFullyDefined ?
                constant_op.constant(s_tensor_shape.dims, dtype: dtypes.int32) :
                null;

            if (@static)
            {
                if (p_static is null) return null;
                var shape = new Shape(p_static).concatenate(s_static);
                throw new NotImplementedException("RNNCell _concat");
            }
            else
            {
                if (p is null || s is null)
                    throw new ValueError($"Provided a prefix or suffix of None: {prefix} and {suffix}");
                return array_ops.concat(new[] { p, s }, 0);
            }
        }

        public static Shape _concat(int[] prefix, int suffix, bool @static = false)
        {
            var p = new Shape(prefix);
            var p_static = prefix;
            var p_tensor = p.IsFullyDefined ? constant_op.constant(p, dtype: dtypes.int32) : null;

            var s_tensor_shape = new Shape(suffix);
            var s_static = s_tensor_shape.ndim > -1 ?
                s_tensor_shape.dims :
                null;
            var s_tensor = s_tensor_shape.IsFullyDefined ?
                constant_op.constant(s_tensor_shape.dims, dtype: dtypes.int32) :
                null;

            if (@static)
            {
                if (p_static is null) return null;
                var shape = new Shape(p_static).concatenate(s_static);
                return shape;
            }
            else
            {
                if (p is null || s_tensor is null)
                    throw new ValueError($"Provided a prefix or suffix of None: {prefix} and {suffix}");
                // return array_ops.concat(new[] { p_tensor, s_tensor }, 0);
                throw new NotImplementedException("");
            }
        }
    }
}
