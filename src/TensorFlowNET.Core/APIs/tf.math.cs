using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public static partial class tf
    {
        public static Tensor abs(Tensor x, string name = null)
            => math_ops.abs(x, name);

        /// <summary>
        /// Computes acos of x element-wise.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor acos(Tensor x, string name = null)
            => gen_math_ops.acos(x, name);

        /// <summary>
        /// Computes asin of x element-wise.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor asin(Tensor x, string name = null)
            => gen_math_ops.asin(x, name);

        public static Tensor add(Tensor a, Tensor b) 
            => gen_math_ops.add(a, b);

        public static Tensor arg_max(Tensor input, int dimension, TF_DataType output_type = TF_DataType.TF_INT64, string name = null)
            => gen_math_ops.arg_max(input, dimension, output_type: output_type, name: name);

        public static Tensor arg_min(Tensor input, int dimension, TF_DataType output_type = TF_DataType.TF_INT64, string name = null)
            => gen_math_ops.arg_min(input, dimension, output_type: output_type, name: name);

        public static Tensor sub(Tensor a, Tensor b) 
            => gen_math_ops.sub(a, b);

        public static Tensor sqrt(Tensor a, string name = null) 
            => gen_math_ops.sqrt(a, name);

        public static Tensor subtract<T>(Tensor x, T[] y, string name = null) where T : struct
            => gen_math_ops.sub(x, ops.convert_to_tensor(y, dtype: x.dtype.as_base_dtype(), name: "y"), name);

        public static Tensor log(Tensor x, string name = null)
            => gen_math_ops.log(x, name);

        public static Tensor equal(Tensor x, Tensor y, string name = null)
            => gen_math_ops.equal(x, y, name);

        public static Tensor multiply(Tensor x, Tensor y) 
            => gen_math_ops.mul(x, y);

        public static Tensor negative(Tensor x, string name = null)
            => gen_math_ops.neg(x, name);

        public static Tensor divide<T>(Tensor x, T[] y, string name = null) where T : struct
            => x / ops.convert_to_tensor(y, dtype: x.dtype.as_base_dtype(), name: "y");

        public static Tensor pow<T1, T2>(T1 x, T2 y) 
            => gen_math_ops.pow(x, y);

        /// <summary>
        /// Computes the sum of elements across dimensions of a tensor.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="axis"></param>
        /// <returns></returns>
        public static Tensor reduce_sum(Tensor input, int? axis = null, int? reduction_indices = null)
        {
            if(!axis.HasValue && reduction_indices.HasValue)
                return math_ops.reduce_sum(input, reduction_indices.Value);
            return math_ops.reduce_sum(input);
        }

        public static Tensor reduce_mean(Tensor input_tensor, int[] axis = null, bool keepdims = false, string name = null, int? reduction_indices = null)
            => math_ops.reduce_mean(input_tensor, axis: axis, keepdims: keepdims, name: name, reduction_indices: reduction_indices);

        public static Tensor cast(Tensor x, TF_DataType dtype = TF_DataType.DtInvalid, string name = null) 
            => math_ops.cast(x, dtype, name);

        public static Tensor argmax(Tensor input, int axis = -1, string name = null, int? dimension = null, TF_DataType output_type = TF_DataType.TF_INT64)
            => gen_math_ops.arg_max(input, axis, name: name, output_type: output_type);
    }
}
