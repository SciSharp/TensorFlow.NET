using System;
using System.Collections;
using System.Collections.Generic;
using System.Numerics;
using System.Text;

namespace Tensorflow.NumPy
{
    public partial class np
    {
        [AutoNumPy]
        public static NDArray reshape(NDArray x1, Shape newshape) => x1.reshape(newshape);

        [AutoNumPy]
        public static NDArray squeeze(NDArray x1, Axis? axis = null) => new NDArray(array_ops.squeeze(x1, axis));
    }
}
