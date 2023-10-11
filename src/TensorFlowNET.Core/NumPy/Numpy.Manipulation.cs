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
        public static NDArray concatenate((NDArray, NDArray) tuple, int axis = 0)
            => new NDArray(array_ops.concat(new[] { tuple.Item1, tuple.Item2 }, axis));

        [AutoNumPy]
        public static NDArray concatenate(NDArray[] arrays, int axis = 0) => new NDArray(array_ops.concat(arrays, axis));

        [AutoNumPy]
        public static NDArray dstack(params NDArray[] tup) => throw new NotImplementedException("");

        [AutoNumPy]
        public static NDArray expand_dims(NDArray a, Axis? axis = null) => new NDArray(array_ops.expand_dims(a, axis: axis ?? -1));

        [AutoNumPy]
        public static NDArray reshape(NDArray x1, Shape newshape) => x1.reshape(newshape);

        [AutoNumPy]
        public static NDArray squeeze(NDArray x1, Axis? axis = null) => new NDArray(array_ops.squeeze(x1, axis));

        [AutoNumPy]
        public static NDArray stack(params NDArray[] arrays) => new NDArray(array_ops.stack(arrays));

        [AutoNumPy]
        public static NDArray stack(NDArray[] arrays, int axis = 0) => new NDArray(array_ops.stack(arrays, axis));
        
        [AutoNumPy]
        public static NDArray stack((NDArray, NDArray) tuple, int axis = 0) => new NDArray(array_ops.stack(new[] { tuple.Item1, tuple.Item2 }, axis));

        [AutoNumPy]
        public static NDArray stack((NDArray, NDArray, NDArray) tuple, int axis = 0) => new NDArray(array_ops.stack(new[] { tuple.Item1, tuple.Item2, tuple.Item3 }, axis));

        [AutoNumPy]
        public static NDArray moveaxis(NDArray array, Axis source, Axis destination) => new NDArray(array_ops.moveaxis(array, source, destination));
    }
}
