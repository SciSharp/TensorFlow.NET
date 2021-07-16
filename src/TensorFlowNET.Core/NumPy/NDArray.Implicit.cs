using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.NumPy
{
    public partial class NDArray
    {
        public void Deconstruct(out byte blue, out byte green, out byte red)
        {
            var data = ToArray<byte>();
            blue = data[0];
            green = data[1];
            red = data[2];
        }

        public static implicit operator NDArray(Array array)
            => new NDArray(array);

        public unsafe static implicit operator bool(NDArray nd)
            => *(bool*)nd.data;

        public unsafe static implicit operator byte(NDArray nd)
            => *(byte*)nd.data;

        public unsafe static implicit operator int(NDArray nd)
            => *(int*)nd.data;

        public unsafe static implicit operator long(NDArray nd)
            => *(long*)nd.data;

        public unsafe static implicit operator float(NDArray nd)
            => *(float*)nd.data;

        public unsafe static implicit operator double(NDArray nd)
            => *(double*)nd.data;

        public static implicit operator NDArray(bool value)
            => new NDArray(value);

        public static implicit operator NDArray(int value)
            => new NDArray(value);

        public static implicit operator NDArray(float value)
            => new NDArray(value);

        public static implicit operator NDArray(double value)
            => new NDArray(value);

        public static implicit operator Tensor(NDArray nd)
            => nd?._tensor;

        public static implicit operator NDArray(Tensor tensor)
            => new NDArray(tensor);
    }
}
