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
        public static NDArray cos(NDArray x) => new NDArray(math_ops.cos(x));

        [AutoNumPy]
        public static NDArray exp(NDArray x) => new NDArray(tf.exp(x));

        [AutoNumPy]
        public static NDArray floor(NDArray x) => new NDArray(math_ops.floor(x));

        [AutoNumPy]
        public static NDArray log(NDArray x) => new NDArray(tf.log(x));

        [AutoNumPy]
        public static NDArray mean(NDArray x) => new NDArray(math_ops.reduce_mean(x));

        [AutoNumPy]
        public static NDArray multiply(NDArray x1, NDArray x2) => new NDArray(tf.multiply(x1, x2));

        [AutoNumPy]
        //public static NDArray maximum(NDArray x1, NDArray x2) => new NDArray(tf.maximum(x1, x2));
        public static NDArray maximum(NDArray x1, NDArray x2, int? axis = null)
        {
            var maxValues = tf.maximum(x1, x2);
            if (axis.HasValue)
            {
                maxValues = tf.reduce_max(maxValues, axis: axis.Value);
            }
            return new NDArray(maxValues);
        }

        [AutoNumPy]
        public static NDArray minimum(NDArray x1, NDArray x2) => new NDArray(tf.minimum(x1, x2));

        [AutoNumPy]
        public static NDArray prod(NDArray array, Axis? axis = null, Type? dtype = null, bool keepdims = false)
            => new NDArray(tf.reduce_prod(array, axis: axis));

        [AutoNumPy]
        public static NDArray prod<T>(params T[] array) where T : unmanaged
            => new NDArray(tf.reduce_prod(new NDArray(array)));
        [AutoNumPy]
        public static NDArray dot(NDArray x1, NDArray x2, NDArray? axes = null, string? name = null)
        {
            //if axes mentioned
            if (axes != null)
            {
                return new NDArray(tf.dot_prod(x1, x2, axes, name));
            }
            if (x1.shape.ndim > 1)
            {
                x1 = GetFlattenArray(x1);
            }
            if (x2.shape.ndim > 1)
            {
                x2 = GetFlattenArray(x2);
            }
            //if axes not mentioned, default 0,0
            return new NDArray(tf.dot_prod(x1, x2, axes: new int[] { 0, 0 }, name));

        }
        [AutoNumPy]
        public static NDArray power(NDArray x, NDArray y) => new NDArray(tf.pow(x, y));
        [AutoNumPy]
        public static NDArray square(NDArray x) => new NDArray(tf.square(x));

        [AutoNumPy]
        public static NDArray sin(NDArray x) => new NDArray(math_ops.sin(x));

        [AutoNumPy]
        public static NDArray sqrt(NDArray x) => new NDArray(tf.sqrt(x));

        [AutoNumPy]
        public static NDArray sum(NDArray x1, Axis? axis = null) => new NDArray(tf.math.sum(x1, axis));

        [AutoNumPy]
        public static NDArray add(NDArray x, NDArray y) => new NDArray(math_ops.add(x, y));

        [AutoNumPy]
        public static NDArray greater(NDArray x, NDArray y) => new NDArray(tf.greater(x, y));

        [AutoNumPy]
        public static NDArray less(NDArray x, NDArray y) => new NDArray(tf.less(x, y));
    }
}
