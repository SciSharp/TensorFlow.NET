using System;
using System.Collections;
using System.Collections.Generic;
using System.Numerics;
using System.Text;

namespace Tensorflow.NumPy
{
    public partial class np
    {
        public static NDArray squeeze(NDArray x1, Axis? axis = null)
            => new NDArray(array_ops.squeeze(x1, axis));
    }
}
