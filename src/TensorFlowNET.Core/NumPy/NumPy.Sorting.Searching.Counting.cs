using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Numerics;
using System.Text;

namespace Tensorflow.NumPy
{
    public partial class np
    {
        [AutoNumPy]
        public static NDArray argmax(NDArray a, Axis? axis = null)
            => new NDArray(math_ops.argmax(a, axis ?? 0));

        [AutoNumPy]
        public static NDArray argmin(NDArray a, Axis? axis = null)
            => new NDArray(math_ops.argmin(a, axis ?? 0));

        [AutoNumPy]
        public static NDArray argsort(NDArray a, Axis? axis = null)
            => new NDArray(sort_ops.argsort(a, axis: axis ?? -1));

        [AutoNumPy]
        public static (NDArray, NDArray) unique(NDArray a)
        {
            var(u, indice) = array_ops.unique(a);
            return (new NDArray(u), new NDArray(indice));
        }

        [AutoNumPy]
        public static void shuffle(NDArray x) => np.random.shuffle(x);

        /// <summary>
        /// Sorts a ndarray
        /// </summary>
        /// <param name="values"></param>
        /// <param name="axis">
        ///     The axis along which to sort. The default is -1, which sorts the last axis.
        /// </param>
        /// <param name="direction">
        ///     The direction in which to sort the values (`'ASCENDING'` or `'DESCENDING'`) 
        /// </param>
        /// <returns>
        ///     A `NDArray` with the same dtype and shape as `values`, with the elements sorted along the given `axis`.
        /// </returns>
        [AutoNumPy]
        public static NDArray sort(NDArray values, Axis? axis = null, string direction = "ASCENDING")
            => new NDArray(sort_ops.sort(values, axis: axis ?? -1, direction: direction));
    }
}
