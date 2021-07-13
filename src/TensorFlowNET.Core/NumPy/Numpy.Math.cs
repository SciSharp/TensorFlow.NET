using System;
using System.Collections;
using System.Collections.Generic;
using System.Numerics;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow.NumPy
{
    public partial class np
    {
        public static NDArray log(NDArray x)
            => tf.log(x);

        public static NDArray prod(NDArray array, Axis? axis = null, Type? dtype = null, bool keepdims = false)
            => tf.reduce_prod(array, axis: axis);

        public static NDArray prod<T>(params T[] array) where T : unmanaged
            => tf.reduce_prod(ops.convert_to_tensor(array));

        public static NDArray multiply(NDArray x1,  NDArray x2)
            => tf.multiply(x1, x2);

        public static NDArray sum(NDArray x1, Axis? axis = null)
            => tf.math.sum(x1, axis);
    }
}
