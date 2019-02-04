using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class math_ops
    {
        /// <summary>
        /// Helper function for reduction ops.
        /// </summary>
        /// <param name="input_shape">1-D Tensor, the shape of the Tensor being reduced.</param>
        /// <param name="axes">1-D Tensor, the reduction axes.</param>
        /// <returns>A 1-D Tensor, the output shape as if keepdims were set to True.</returns>
        public static Tensor reduced_shape(Tensor input_shape, Tensor axes)
        {
            input_shape = to_int32(input_shape);
            axes = to_int32(axes);

            var input_rank = array_ops.size(input_shape);
            axes = (axes + input_rank) % input_rank;

            return null;
        }

        /// <summary>
        /// Casts a tensor to type `int32`.
        /// </summary>
        /// <param name="x">A `Tensor` or `SparseTensor` or `IndexedSlices`.</param>
        /// <param name="name">A name for the operation (optional).</param>
        /// <returns>A `Tensor` or `SparseTensor` or `IndexedSlices` with same shape as `x` with type `int32`.</returns>
        private static Tensor to_int32(Tensor x, string name = "ToInt32")
        {
            return __case__(x, TF_DataType.TF_INT32, name: name);
        }

        /// <summary>
        /// Casts a tensor to a new type.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="dtype"></param>
        /// <param name="name"></param>
        /// <returns>A `Tensor` or `SparseTensor` or `IndexedSlices` with same shape as `x` and same type as `dtype`.</returns>
        public static Tensor __case__(Tensor x, TF_DataType dtype, string name = "")
        {
            var base_type = dtype.as_base_dtype();
            if (x is Tensor && base_type == x.dtype)
                return x;

            // math_ops.py cast
            throw new NotImplementedException();
        }

        public static Tensor reduce_sum(Tensor input_tensor, Tensor axis = null, bool keepdims = false)
        {
            var r = _ReductionDims(input_tensor, axis);
            var m = gen_math_ops.sum(input_tensor, r);
            return _may_reduce_to_scalar(keepdims, m);
        }

        private static Tensor _may_reduce_to_scalar(bool keepdims, Tensor output)
        {
            output.shape = new long[0];
            return output;
        }

        private static Tensor _ReductionDims(Tensor x, Tensor axis)
        {
            if (axis != null)
            {
                return axis;
            }
            else
            {
                var rank = array_ops.rank(x);
                return range(0, rank, 1);
            }
        }

        public static Tensor range(object start, Tensor limit = null, object delta = null, TF_DataType dtype = TF_DataType.DtInvalid, string name = "range" )
        {
            return Python.with<ops.name_scope, Tensor>(new ops.name_scope(name, "Range", new object[] { start, limit, delta }), scope =>
            {
                name = scope;
                var start1 = ops.convert_to_tensor(start, name: "start");
                var limit1 = ops.convert_to_tensor(limit, name: "limit");
                var delta1 = ops.convert_to_tensor(delta, name: "delta");

                return gen_math_ops.range(start1, limit1, delta1, name);
            });
        }

        public static Tensor rank_internal(Tensor input, string name = "", bool optimize = true)
        {
            return Python.with<ops.name_scope, Tensor>(new ops.name_scope(name, "Rank", new List<Tensor> { input }), scope =>
            {
                name = scope;
                var input_tensor = ops.convert_to_tensor(input);
                var input_shape = tensor_util.to_shape(input_tensor.shape);
                if (optimize && input_shape.NDim == null)
                    return constant_op.constant(input_shape.NDim);
                else
                    return gen_array_ops.rank(input, name);
            });
        }

        public static Tensor matmul(Tensor a, Tensor b,
            bool transpose_a = false, bool transpose_b = false,
            bool adjoint_a = false, bool adjoint_b = false,
            bool a_is_sparse = false, bool b_is_sparse = false,
            string name = "")
        {
            Tensor result = null;

            Python.with<ops.name_scope>(new ops.name_scope(name, "MatMul", new Tensor[] { a, b }), scope =>
            {
                name = scope;

                if (transpose_a && adjoint_a)
                    throw new ValueError("Only one of transpose_a and adjoint_a can be True.");
                if (transpose_b && adjoint_b)
                    throw new ValueError("Only one of transpose_b and adjoint_b can be True.");

                a = ops.convert_to_tensor(a, name: "a");
                b = ops.convert_to_tensor(b, name: "b");

                result = gen_math_ops.mat_mul(a, b, transpose_a, transpose_b, name);
            });

            return result;
        }

        /// <summary>
        /// Returns the complex conjugate of a complex number.
        /// </summary>
        /// <param name="x">`Tensor` to conjugate.  Must have numeric or variant type.</param>
        /// <param name="name">A name for the operation (optional).</param>
        /// <returns>A `Tensor` that is the conjugate of `x` (with the same type).</returns>
        public static Tensor conj(Tensor x, string name = "")
        {
            var dt = x.dtype;
            if (dt.is_floating() || dt.is_integer())
                return x;

            return Python.with<ops.name_scope, Tensor>(new ops.name_scope(name, "Conj", new List<Tensor> { x }), scope =>
            {

                return x;
            });
        }
    }
}
