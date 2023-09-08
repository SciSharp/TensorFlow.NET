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
        public void shuffle(NDArray x, int? seed = null)
        {
            var y = random_ops.random_shuffle(x, seed);
            Marshal.Copy(y.BufferToArray(), 0, x.TensorDataPointer, (int)x.bytesize);
        }

        public NDArray random(Shape size)
            => uniform(low: 0, high: 1, size: size);

        [AutoNumPy]
        public NDArray randint(int low, int? high = null, Shape? size = null, TF_DataType dtype = TF_DataType.TF_INT32)
        {
            if(high == null)
            {
                high = low;
                low = 0;
            }
            size = size ?? Shape.Scalar;
            var tensor = random_ops.random_uniform_int(shape: size, minval: low, maxval: (int)high);
            return new NDArray(tensor);
        }

        [AutoNumPy]
        public NDArray randn(params int[] shape)
            => new NDArray(random_ops.random_normal(shape ?? Shape.Scalar));

        [AutoNumPy]
        public NDArray normal(float loc = 0.0f, float scale = 1.0f, Shape? size = null)
            => new NDArray(random_ops.random_normal(size ?? Shape.Scalar, mean: loc, stddev: scale));

        [AutoNumPy]
        public NDArray uniform(float low = 0.0f, float high = 1.0f, Shape? size = null)
            => new NDArray(random_ops.random_uniform(size ?? Shape.Scalar, low, high));
    }
}
