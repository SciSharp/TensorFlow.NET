using System;
using System.Collections;
using System.Collections.Generic;
using System.Numerics;
using System.Text;

namespace Tensorflow.NumPy
{
    public partial class np
    {
        public static NDArray argmax(NDArray a, Axis axis = null)
            => new NDArray(math_ops.argmax(a, axis));

        public static NDArray argsort(NDArray a, Axis axis = null)
            => new NDArray(math_ops.argmax(a, axis ?? -1));

        public static NDArray unique(NDArray a)
            => throw new NotImplementedException("");
    }
}
