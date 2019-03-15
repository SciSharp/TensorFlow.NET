using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public static partial class tf
    {
        /// <summary>
        /// Inserts a dimension of 1 into a tensor's shape.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="axis"></param>
        /// <param name="name"></param>
        /// <param name="dim"></param>
        /// <returns>
        /// A `Tensor` with the same data as `input`, but its shape has an additional
        /// dimension of size 1 added.
        /// </returns>
        public static Tensor expand_dims(Tensor input, int axis = -1, string name = null, int dim = -1)
            => array_ops.expand_dims(input, axis, name, dim);

        /// <summary>
        /// Transposes `a`. Permutes the dimensions according to `perm`.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="perm"></param>
        /// <param name="name"></param>
        /// <param name="conjugate"></param>
        /// <returns></returns>
        public static Tensor transpose<T1, T2>(T1 a, T2 perm, string name = "transpose", bool conjugate = false)
            => array_ops.transpose(a, perm, name, conjugate);

        public static Tensor squeeze(Tensor input, int[] axis = null, string name = null, int squeeze_dims = -1)
            => gen_array_ops.squeeze(input, axis, name);

        public static Tensor one_hot(Tensor indices, int depth,
            Tensor on_value = null,
            Tensor off_value = null,
            TF_DataType dtype = TF_DataType.DtInvalid,
            int axis = -1,
            string name = null) => array_ops.one_hot(indices, depth, dtype: dtype, axis: axis, name: name);        
    }
}
