/*****************************************************************************
   Copyright 2021 Haiping Chen. All Rights Reserved.

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
using static Tensorflow.Binding;

namespace Tensorflow
{
    /// <summary>
    /// Represents a sparse tensor.
    /// </summary>
    public class SparseTensor : CompositeTensor
    {
        public Tensor indices;

        public Tensor values;

        public Tensor dense_shape;

        public SparseTensor(Tensor indices, Tensor values, Tensor dense_shape)
        {
            this.indices = indices;
            this.values = values;
            this.dense_shape = dense_shape;
            _init();
        }

        public SparseTensor(long[,] indices_, Array values_, long[] dense_shape_)
        {
            tf_with(ops.name_scope(null, "SparseTensor", new { }), delegate
            {
                indices = ops.convert_to_tensor(
                    indices_, name: "indices", dtype: dtypes.int64);
                values = ops.convert_to_tensor(values_, name: "values");
                dense_shape = ops.convert_to_tensor(
                    dense_shape_, name: "dense_shape", dtype: dtypes.int64);
            });
            _init();
        }

        void _init()
        {
            var indices_shape = indices.TensorShape.with_rank(2);
            var values_shape = values.TensorShape.with_rank(1);
            var dense_shape_shape = dense_shape.TensorShape.with_rank(1);

            indices_shape["0"].merge_with(values_shape[0]);
            indices_shape["1"].merge_with(dense_shape_shape[0]);
        }

        public static implicit operator Tensor(SparseTensor indexedSlices)
        {
            return indexedSlices.values;
        }

        public static implicit operator SparseTensor(Tensor tensor)
        {
            return tensor.Tag as SparseTensor;
        }
    }
}
