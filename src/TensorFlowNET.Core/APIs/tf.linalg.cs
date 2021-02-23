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
using static Tensorflow.Binding;

namespace Tensorflow
{
    public partial class tensorflow
    {
        public LinalgApi linalg { get; } = new LinalgApi();

        public class LinalgApi
        {
            linalg_ops ops = new linalg_ops();

            public Tensor eye(int num_rows,
                int num_columns = -1,
                TensorShape batch_shape = null,
                TF_DataType dtype = TF_DataType.TF_FLOAT,
                string name = null)
                => ops.eye(num_rows, num_columns: num_columns, batch_shape: batch_shape, dtype: dtype, name: name);

            public Tensor diag(Tensor diagonal, string name = null)
                => gen_array_ops.diag(diagonal, name: name);

            public Tensor matmul(Tensor a, Tensor b)
                => math_ops.matmul(a, b);

            public Tensor batch_matmul(Tensor x, Tensor y, bool adj_x = false, bool adj_y = false, string name = null)
                => math_ops.batch_matmul(x, y, adj_x: adj_x, adj_y: adj_y, name: name);
        }

        public Tensor diag(Tensor diagonal, string name = null)
            => gen_array_ops.diag(diagonal, name: name);

        public Tensor matmul(Tensor a, Tensor b)
            => math_ops.matmul(a, b);

        /// <summary>
        /// Multiply slices of the two matrices "x" and "y".
        /// </summary>
        /// <remarks>
        /// The `BatchMatMul` operation is embedded into the
        /// `MatMul` operation on the DLL side. However the expected
        /// attributes are not the same, hence we need to expose this
        /// method to have the right args list on the `_apply_op_helper`
        /// function.
        ///
        /// For each rank > 2 the first rank - 2 dimensions are considered
        /// as fixed, and have to be consistent across the two matrices. A
        /// common matrix multiplication is then applied over the residual
        /// 2 dimensions.
        ///
        /// e.g.
        ///     x is (3, 6, 12); y is (3, 12, 6)
        ///     batch_matmul(x, y) ==> (3, 6, 6)
        /// </remarks>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="adj_x"></param>
        /// <param name="adj_y"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor batch_matmul(Tensor x, Tensor y, bool adj_x = false, bool adj_y = false, string name = null)
            => math_ops.batch_matmul(x, y, adj_x: adj_x, adj_y: adj_y, name: name);
    }
}
