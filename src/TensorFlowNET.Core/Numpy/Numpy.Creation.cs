using System;
using System.Collections;
using System.Collections.Generic;
using System.Numerics;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow.Numpy
{
    public partial class np
    {
        public static NDArray array(object data)
            => throw new NotImplementedException("");

        public static NDArray array(Array data)
            => new NDArray(tf.constant(data));

        public static NDArray array<T>(params T[] data)
            where T : unmanaged
            => new NDArray(tf.constant(data));

        public static NDArray array(params float[] data)
            => throw new NotImplementedException("");

        public static NDArray arange<T>(T end)
            where T : unmanaged
            => new NDArray(tf.range(default(T), limit: end));

        public static NDArray arange<T>(T start, T? end = null, T? step = null)
            where T : unmanaged
            => new NDArray(tf.range(start, limit: end, delta: step));

        public static NDArray empty(Shape shape, NumpyDType dtype = NumpyDType.Double)
            => new NDArray(tf.zeros(shape, dtype: dtype.as_tf_dtype()));

        public static NDArray ones(Shape shape, NumpyDType dtype = NumpyDType.Double)
            => new NDArray(tf.ones(shape, dtype: dtype.as_tf_dtype()));

        public static NDArray ones_like(NDArray a, Type dtype = null)
            => throw new NotImplementedException("");

        public static NDArray zeros(Shape shape, NumpyDType dtype = NumpyDType.Double)
            => new NDArray(tf.zeros(shape, dtype: dtype.as_tf_dtype()));

        public static NDArray full<T>(Shape shape, T fill_value)
            => new NDArray(tf.fill(tf.constant(shape), fill_value));
    }
}
