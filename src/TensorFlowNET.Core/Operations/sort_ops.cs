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
    public class sort_ops
    {
        public static Tensor argsort(Tensor values, Axis axis = null, string direction = "ASCENDING", bool stable = false, string name = null)
        {
            axis = axis ?? new Axis(-1);
            var k = array_ops.shape(values)[axis];
            values = -values;
            var static_rank = values.shape.ndim;
            var top_k_input = values;
            if (axis == -1 || axis + 1 == values.shape.ndim)
            {
            }
            else
            {
                if (axis == 0 && static_rank == 2)
                    top_k_input = array_ops.transpose(values, new[] { 1, 0 });
                else
                    throw new NotImplementedException("");
            }

            var (_, indices) = tf.Context.ExecuteOp("TopKV2", name,
                new ExecuteOpArgs(top_k_input, k).SetAttributes(new
                {
                    sorted = true
                }));
            return indices.Single;
        }

        public static Tensor sort(Tensor values, Axis axis, string direction = "ASCENDING", string? name = null)
        {
            var k = array_ops.shape(values)[axis];
            values = -values;
            var static_rank = values.shape.ndim;
            var top_k_input = values;
            if (axis == -1 || axis + 1 == values.shape.ndim)
            {
            }
            else
            {
                if (axis == 0 && static_rank == 2)
                    top_k_input = array_ops.transpose(values, new[] { 1, 0 });
                else
                    throw new NotImplementedException("");
            }

            (values, _) = tf.Context.ExecuteOp("TopKV2", name,
                new ExecuteOpArgs(top_k_input, k).SetAttributes(new
                {
                    sorted = true
                }));
            return -values;
        }

        public Tensor matrix_inverse(Tensor input, bool adjoint = false, string name = null)
            => tf.Context.ExecuteOp("MatrixInverse", name,
                new ExecuteOpArgs(input).SetAttributes(new
                {
                    adjoint
                }));
    }
}
