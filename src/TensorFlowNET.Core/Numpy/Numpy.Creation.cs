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
        public static NDArray array(Array data)
            => new NDArray(data);

        public static NDArray array<T>(params T[] data)
            where T : unmanaged
            => new NDArray(data);

        public static NDArray arange<T>(T end)
            where T : unmanaged
            => new NDArray(tf.range(default(T), limit: end));

        public static NDArray arange<T>(T start, T? end = null, T? step = null)
            where T : unmanaged
            => new NDArray(tf.range(start, limit: end, delta: step));

        public static NDArray empty(Shape shape, TF_DataType dtype = TF_DataType.TF_DOUBLE)
            => new NDArray(tf.zeros(shape, dtype: dtype));

        public static NDArray eye(int N, int? M = null, int k = 0, TF_DataType dtype = TF_DataType.TF_DOUBLE)
            => tf.numpy.eye(N, M: M, k: k, dtype: dtype);

        public static NDArray full<T>(Shape shape, T fill_value)
            => new NDArray(tf.fill(tf.constant(shape), fill_value));

        public static NDArray linspace<T>(T start, T stop, int num = 50, bool endpoint = true, bool retstep = false,
            TF_DataType dtype = TF_DataType.TF_DOUBLE, int axis = 0) where T : unmanaged
            => tf.numpy.linspace(start, stop, num: num, endpoint: endpoint, retstep: retstep, dtype: dtype, axis: axis);

        public static (NDArray, NDArray) meshgrid<T>(T x, T y, bool copy = true, bool sparse = false)
            => tf.numpy.meshgrid(new[] { x, y }, copy: copy, sparse: sparse);

        public static NDArray ones(Shape shape, TF_DataType dtype = TF_DataType.TF_DOUBLE)
            => new NDArray(tf.ones(shape, dtype: dtype));

        public static NDArray ones_like(NDArray a, Type dtype = null)
            => throw new NotImplementedException("");

        public static NDArray zeros(Shape shape, TF_DataType dtype = TF_DataType.TF_DOUBLE)
            => new NDArray(tf.zeros(shape, dtype: dtype));
    }
}
