using System;
using System.Linq;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class linalg_ops
    {
        public Tensor eye(int num_rows,
            int num_columns = -1,
            Shape batch_shape = null,
            TF_DataType dtype = TF_DataType.TF_DOUBLE,
            string name = null)
        {
            return tf_with(ops.name_scope(name, default_name: "eye", new { num_rows, num_columns, batch_shape }), scope =>
            {
                if (num_columns == -1)
                    num_columns = num_rows;

                bool is_square = num_columns == num_rows;
                var diag_size = Math.Min(num_rows, num_columns);
                if (batch_shape == null)
                    batch_shape = new Shape(new int[0]);
                var batch_shape_tensor = ops.convert_to_tensor(batch_shape, dtype: tf.int32, name: "shape");
                var diag_shape = array_ops.concat(new[] { batch_shape_tensor, tf.constant(new int[] { diag_size }) }, axis: 0);

                Tensor shape = null;
                if (!is_square)
                    shape = array_ops.concat(new[] { batch_shape_tensor, tf.constant(new int[] { num_rows, num_columns }) }, axis: 0);

                var diag_ones = array_ops.ones(diag_shape, dtype: dtype);
                if (is_square)
                    return array_ops.matrix_diag(diag_ones);
                else
                {
                    var zero_matrix = array_ops.zeros(shape, dtype: dtype);
                    return array_ops.matrix_set_diag(zero_matrix, diag_ones);
                }
            });
        }

        public Tensor matrix_inverse(Tensor input, bool adjoint = false, string name = null)
            => tf.Context.ExecuteOp("MatrixInverse", name,
                new ExecuteOpArgs(input).SetAttributes(new
                {
                    adjoint
                }));

        public Tensor matrix_solve_ls(Tensor matrix, Tensor rhs,
            Tensor l2_regularizer = null, bool fast = true, string name = null)
        {
            return _composite_impl(matrix, rhs, l2_regularizer: l2_regularizer);
        }

        public Tensor norm(Tensor tensor, string ord = "euclidean", Axis axis = null, string name = null, bool keepdims = true)
        {
            var is_matrix_norm = axis != null && len(axis) == 2;
            return tf_with(ops.name_scope(name, default_name: "norm", tensor), scope =>
            {
                if (is_matrix_norm)
                    throw new NotImplementedException("");
                var result = math_ops.sqrt(math_ops.reduce_sum(tensor * math_ops.conj(tensor), axis, keepdims: true));

                if(!keepdims)
                    result = array_ops.squeeze(result, axis);
                return result;
            });
        }

        Tensor _composite_impl(Tensor matrix, Tensor rhs, Tensor l2_regularizer = null)
        {
            Shape matrix_shape = matrix.shape.dims.Skip(matrix.shape.ndim - 2).ToArray();
            if (matrix_shape.IsFullyDefined)
            {
                if (matrix_shape[-2] >= matrix_shape[-1])
                    return _overdetermined(matrix, rhs, l2_regularizer);
                else
                    return _underdetermined(matrix, rhs, l2_regularizer);
            }

            throw new NotImplementedException("");
        }

        Tensor _overdetermined(Tensor matrix, Tensor rhs, Tensor l2_regularizer = null)
        {
            var chol = _RegularizedGramianCholesky(matrix, l2_regularizer: l2_regularizer, first_kind: true);
            return cholesky_solve(chol, math_ops.matmul(matrix, rhs, adjoint_a: true));
        }

        Tensor _underdetermined(Tensor matrix, Tensor rhs, Tensor l2_regularizer = null)
        {
            var chol = _RegularizedGramianCholesky(matrix, l2_regularizer: l2_regularizer, first_kind: false);
            return math_ops.matmul(matrix, cholesky_solve(chol, rhs), adjoint_a: true);
        }

        Tensor _RegularizedGramianCholesky(Tensor matrix, Tensor l2_regularizer, bool first_kind)
        {
            var gramian = math_ops.matmul(matrix, matrix, adjoint_a: first_kind, adjoint_b: !first_kind);

            if (l2_regularizer != null)
            {
                var matrix_shape = array_ops.shape(matrix);
                var batch_shape = matrix_shape[":-2"];
                var small_dim = first_kind ? matrix_shape[-1] : matrix_shape[-2];
                var identity = eye(small_dim.numpy(), batch_shape: batch_shape.shape, dtype: matrix.dtype);
                var small_dim_static = matrix.shape[first_kind ? -1 : -2];
                identity.shape = matrix.shape.dims.Take(matrix.shape.ndim - 2).ToArray().concat(new[] { small_dim_static, small_dim_static });
                gramian += l2_regularizer * identity;
            }

            return cholesky(gramian);
        }

        public Tensor cholesky(Tensor input, string name = null)
            => tf.Context.ExecuteOp("Cholesky", name, new ExecuteOpArgs(input));

        public Tensor cholesky_solve(Tensor chol, Tensor rhs, string name = null)
            => tf_with(ops.name_scope(name, default_name: "eye", new { chol, rhs }), scope =>
            {
                var y = matrix_triangular_solve(chol, rhs, adjoint: false, lower: true);
                var x = matrix_triangular_solve(chol, y, adjoint: true, lower: true);
                return x;
            });

        public Tensor matrix_triangular_solve(Tensor matrix, Tensor rhs, bool lower = true, bool adjoint = false, string name = null)
            => tf.Context.ExecuteOp("MatrixTriangularSolve", name,
                new ExecuteOpArgs(matrix, rhs).SetAttributes(new
                {
                    lower,
                    adjoint
                }));

        public Tensors qr(Tensor input, bool full_matrices = false, string name = null)
            => tf.Context.ExecuteOp("Qr", name,
                new ExecuteOpArgs(input).SetAttributes(new
                {
                    full_matrices
                }));
    }
}
