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
        public static NDArray concatenate(NDArray[] arrays, int axis = 0) => new NDArray(array_ops.concat(arrays, axis));

        [AutoNumPy]
        public static NDArray dstack(params NDArray[] tup) => throw new NotImplementedException("");

        [AutoNumPy]
        public static NDArray expand_dims(NDArray a, Axis? axis = null) => throw new NotImplementedException("");

        [AutoNumPy]
        public static NDArray reshape(NDArray x1, Shape newshape) => x1.reshape(newshape);

        [AutoNumPy]
        public static NDArray squeeze(NDArray x1, Axis? axis = null) => new NDArray(array_ops.squeeze(x1, axis));

        [AutoNumPy]
        public static NDArray stack(params NDArray[] arrays) => new NDArray(array_ops.stack(arrays));
    }
}
