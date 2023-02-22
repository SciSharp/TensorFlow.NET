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
using Tensorflow.NumPy;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public partial class tensorflow
    {
        public LinalgApi linalg { get; } = new LinalgApi();

        public class LinalgApi
        {
            linalg_ops ops = new linalg_ops();

            public Tensor einsum(string equation, Tensors inputs, string name = null)
                => math_ops.einsum(equation, inputs, name: name);

            public Tensor eye(int num_rows,
                int num_columns = -1,
                Shape batch_shape = null,
                TF_DataType dtype = TF_DataType.TF_DOUBLE,
                string name = null)
                => ops.eye(num_rows, num_columns: num_columns, batch_shape: batch_shape, dtype: dtype, name: name);

            public Tensor diag(Tensor diagonal, string name = null)
                => gen_array_ops.diag(diagonal, name: name);

            public Tensor matmul(Tensor a, Tensor b)
                => math_ops.matmul(a, b);

            public Tensor norm(Tensor a, string ord = "euclidean", Axis axis = null, string name = null)
                => ops.norm(a, ord: ord, axis: axis, name: name);

            public Tensor batch_matmul(Tensor x, Tensor y, bool adj_x = false, bool adj_y = false, string name = null)
                => math_ops.batch_matmul(x, y, adj_x: adj_x, adj_y: adj_y, name: name);

            public Tensor inv(Tensor input, bool adjoint = false, string name = null)
                => ops.matrix_inverse(input, adjoint: adjoint, name: name);

            public Tensor global_norm(Tensor[] t_list, string name = null)
                => clip_ops.global_norm(t_list, name: name);

            public Tensor l2_normalize(Tensor x,
                    int axis = 0,
                    float epsilon = 1e-12f,
                    string name = null)
                => nn_impl.l2_normalize(x, axis: axis, epsilon: constant_op.constant(epsilon), name: name);

            public Tensor lstsq(Tensor matrix, Tensor rhs,
                NDArray l2_regularizer = null, bool fast = true, string name = null)
                => ops.matrix_solve_ls(matrix, rhs, l2_regularizer: l2_regularizer, fast: fast, name: name);

            public Tensors qr(Tensor input, bool full_matrices = true, string name = null)
                => ops.qr(input, full_matrices: full_matrices, name: name);

            public Tensor tensor_diag_part(Tensor input, string name = null)
                => gen_array_ops.diag_part(input, name: name);

            public Tensor tensordot(Tensor x, Tensor y, NDArray axes, string name = null)
                => math_ops.tensordot(x, y, axes, name: name);
        }

        public Tensor diag(Tensor diagonal, string name = null)
            => gen_array_ops.diag(diagonal, name: name);

        public Tensor matmul(Tensor a, Tensor b, bool transpose_a = false, bool transpose_b = false)
            => math_ops.matmul(a, b, transpose_a: transpose_a, transpose_b: transpose_b);

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
