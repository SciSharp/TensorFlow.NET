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
        [AutoNumPy]
        public static NDArray any(NDArray a, Axis axis = null) => throw new NotImplementedException("");
        [AutoNumPy]
        public static NDArray logical_or(NDArray x1, NDArray x2) => new NDArray(tf.logical_or(x1, x2));

        [AutoNumPy]
        public static NDArray logical_and(NDArray x1, NDArray x2) => new NDArray(tf.logical_and(x1, x2));
    }
}
