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
        public static NDArray exp(NDArray x)
            => new NDArray(tf.exp(x));

        public static NDArray log(NDArray x)
            => new NDArray(tf.log(x));

        public static NDArray multiply(NDArray x1, NDArray x2)
            => new NDArray(tf.multiply(x1, x2));

        public static NDArray maximum(NDArray x1, NDArray x2)
            => new NDArray(tf.maximum(x1, x2));

        public static NDArray minimum(NDArray x1, NDArray x2)
            => new NDArray(tf.minimum(x1, x2));

        public static NDArray prod(NDArray array, Axis? axis = null, Type? dtype = null, bool keepdims = false)
            => new NDArray(tf.reduce_prod(array, axis: axis));

        public static NDArray prod<T>(params T[] array) where T : unmanaged
            => new NDArray(tf.reduce_prod(new NDArray(array)));

        public static NDArray sqrt(NDArray x)
            => new NDArray(tf.sqrt(x));

        public static NDArray sum(NDArray x1, Axis? axis = null)
            => new NDArray(tf.math.sum(x1, axis));
    }
}
