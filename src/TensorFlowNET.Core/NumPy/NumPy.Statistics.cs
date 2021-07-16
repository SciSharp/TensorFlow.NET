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
        public static NDArray amin(NDArray x, int axis = 0)
            => tf.arg_min(x, axis);

        public static NDArray amax(NDArray x, int axis = 0)
            => tf.arg_max(x, axis);
    }
}
