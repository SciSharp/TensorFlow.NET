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
        public static NDArray amin(NDArray x, int axis = 0) => new NDArray(tf.min(x, axis));

        [AutoNumPy]
        public static NDArray amax(NDArray x, int axis = 0) => new NDArray(tf.max(x, axis));

        [AutoNumPy]
        public static NDArray average(NDArray a, int axis = -1, NDArray? weights = null, bool returned = false)
            => tf.numpy.average(a, axis: axis, weights: weights, returned: returned);
    }
}
