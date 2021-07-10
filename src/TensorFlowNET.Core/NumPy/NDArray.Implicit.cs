using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.NumPy
{
    public partial class NDArray
    {
        public static implicit operator NDArray(Array array)
            => new NDArray(array);

        public static implicit operator bool(NDArray nd)
            => nd._tensor.ToArray<bool>()[0];

        public static implicit operator byte[](NDArray nd)
            => nd.ToByteArray();

        public static implicit operator int(NDArray nd)
            => nd._tensor.ToArray<int>()[0];

        public static implicit operator float(NDArray nd)
            => nd._tensor.ToArray<float>()[0];

        public static implicit operator double(NDArray nd)
            => nd._tensor.ToArray<double>()[0];

        public static implicit operator NDArray(bool value)
            => new NDArray(value);

        public static implicit operator NDArray(int value)
            => new NDArray(value);

        public static implicit operator NDArray(float value)
            => new NDArray(value);

        public static implicit operator NDArray(double value)
            => new NDArray(value);

        public static implicit operator Tensor(NDArray nd)
            => nd._tensor;

        public static implicit operator NDArray(Tensor tensor)
            => new NDArray(tensor);
    }
}
