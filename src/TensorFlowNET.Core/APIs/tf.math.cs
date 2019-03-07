using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public static partial class tf
    {
        public static Tensor add(Tensor a, Tensor b) => gen_math_ops.add(a, b);

        public static Tensor sub(Tensor a, Tensor b) => gen_math_ops.sub(a, b);

        public static Tensor sqrt(Tensor a, string name = null) => gen_math_ops.sqrt(a, name);

        public static Tensor subtract<T>(Tensor x, T[] y, string name = null) where T : struct
            => gen_math_ops.sub(x, ops.convert_to_tensor(y, dtype: x.dtype.as_base_dtype(), name: "y"), name);

        public static Tensor multiply(Tensor x, Tensor y) => gen_math_ops.mul(x, y);

        public static Tensor divide<T>(Tensor x, T[] y, string name = null) where T : struct
            => x / ops.convert_to_tensor(y, dtype: x.dtype.as_base_dtype(), name: "y");

        public static Tensor pow<T1, T2>(T1 x, T2 y) => gen_math_ops.pow(x, y);

        /// <summary>
        /// Computes the sum of elements across dimensions of a tensor.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="axis"></param>
        /// <returns></returns>
        public static Tensor reduce_sum(Tensor input, int[] axis = null) => math_ops.reduce_sum(input);

        public static Tensor cast(Tensor x, TF_DataType dtype = TF_DataType.DtInvalid, string name = null) 
            => math_ops.cast(x, dtype, name);
    }
}
