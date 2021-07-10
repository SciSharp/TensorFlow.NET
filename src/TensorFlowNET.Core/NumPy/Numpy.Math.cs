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
            => throw new NotImplementedException("");

        public static NDArray prod(NDArray array, Axis? axis = null, Type? dtype = null, bool keepdims = false)
            => tf.reduce_prod(ops.convert_to_tensor(array), axis: axis);

        public static NDArray prod<T>(params T[] array) where T : unmanaged
            => tf.reduce_prod(ops.convert_to_tensor(array));

        public static NDArray multiply(in NDArray x1, in NDArray x2)
            => throw new NotImplementedException("");

        public static NDArray sum(NDArray x1)
            => throw new NotImplementedException("");
    }
}
