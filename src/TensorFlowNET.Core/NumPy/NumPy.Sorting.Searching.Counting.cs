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
        public static NDArray argmax(NDArray a, Axis axis = null)
            => new NDArray(math_ops.argmax(a, axis ?? 0));

        [AutoNumPy]
        public static NDArray argsort(NDArray a, Axis axis = null)
            => new NDArray(sort_ops.argsort(a, axis: axis ?? -1));

        [AutoNumPy]
        public static (NDArray, NDArray) unique(NDArray a)
        {
            var(u, indice) = array_ops.unique(a);
            return (new NDArray(u), new NDArray(indice));
        }

        [AutoNumPy]
        public static void shuffle(NDArray x) => np.random.shuffle(x);
    }
}
