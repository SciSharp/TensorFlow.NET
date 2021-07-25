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

using Tensorflow.Operations;
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
            var (_, indices) = tf.Context.ExecuteOp("TopKV2", name,
                new ExecuteOpArgs(values, k).SetAttributes(new
                {
                    sorted = true
                }));
            return indices;
        }

        public Tensor matrix_inverse(Tensor input, bool adjoint = false, string name = null)
            => tf.Context.ExecuteOp("MatrixInverse", name,
                new ExecuteOpArgs(input).SetAttributes(new
                {
                    adjoint
                }));
    }
}
