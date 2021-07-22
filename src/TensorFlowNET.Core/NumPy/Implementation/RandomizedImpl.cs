using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow.NumPy
{
    public class RandomizedImpl
    {
        [AutoNumPy]
        public NDArray permutation(int x) => new NDArray(random_ops.random_shuffle(math_ops.range(0, x)));

        [AutoNumPy]
        public NDArray permutation(NDArray x) => new NDArray(random_ops.random_shuffle(x));

        [AutoNumPy]
        public void shuffle(NDArray x)
        {
            var y = random_ops.random_shuffle(x);
            Marshal.Copy(y.BufferToArray(), 0, x.TensorDataPointer, (int)x.bytesize);
        }

        public NDArray rand(params int[] shape)
            => throw new NotImplementedException("");

        public NDArray randint(int low, int? high = null, Shape size = null, TF_DataType dtype = TF_DataType.TF_INT32)
            => throw new NotImplementedException("");

        public NDArray normal(float loc = 0.0f, float scale = 1.0f, Shape size = null)
            => throw new NotImplementedException("");
    }
}
