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
using Tensorflow.Framework;

namespace Tensorflow
{
    public partial class tensorflow
    {
        public SparseTensor SparseTensor(long[,] indices, Array values, long[] dense_shape)
            => new SparseTensor(indices, values, dense_shape);

        public Tensor sparse_tensor_to_dense(SparseTensor sp_input,
            Array default_value = default,
            bool validate_indices = true,
            string name = null)
            => gen_sparse_ops.sparse_to_dense(sp_input.indices,
                sp_input.dense_shape,
                sp_input.values,
                default_value: default_value,
                validate_indices: validate_indices,
                name: name);

        /// <summary>
        /// Converts a sparse representation into a dense tensor.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="sparse_indices"></param>
        /// <param name="output_shape"></param>
        /// <param name="sparse_values"></param>
        /// <param name="default_value"></param>
        /// <param name="validate_indices"></param>
        /// <param name="name"></param>
        /// <returns>Dense `Tensor` of shape `output_shape`.  Has the same type as `sparse_values`.</returns>
        public Tensor sparse_to_dense<T>(Tensor sparse_indices,
            TensorShape output_shape,
            T sparse_values,
            T default_value = default,
            bool validate_indices = true,
            string name = null)
            => gen_sparse_ops.sparse_to_dense(sparse_indices,
                output_shape,
                sparse_values,
                default_value: default_value,
                validate_indices: validate_indices,
                name: name);
    }
}
