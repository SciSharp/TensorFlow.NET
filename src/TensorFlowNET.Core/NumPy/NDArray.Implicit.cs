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

        public static implicit operator NDArray(int[] array)
            => new NDArray(array);

        public static implicit operator NDArray(byte[] array)
            => new NDArray(array);

        public static implicit operator NDArray(float[] array)
            => new NDArray(array);

        public static implicit operator NDArray(double[] array)
            => new NDArray(array);

        public static implicit operator NDArray(long[] array)
            => new NDArray(array);

        public static implicit operator NDArray(bool[] array)
            => new NDArray(array);

        public static implicit operator NDArray(uint[] array)
            => new NDArray(array);

        public static implicit operator NDArray(ulong[] array)
            => new NDArray(array);

        public static implicit operator NDArray(int[,] array)
            => new NDArray(array);

        public static implicit operator NDArray(byte[,] array)
            => new NDArray(array);

        public static implicit operator NDArray(float[,] array)
            => new NDArray(array);

        public static implicit operator NDArray(double[,] array)
            => new NDArray(array);

        public static implicit operator NDArray(long[,] array)
            => new NDArray(array);

        public static implicit operator NDArray(bool[,] array)
            => new NDArray(array);

        public static implicit operator NDArray(uint[,] array)
            => new NDArray(array);

        public static implicit operator NDArray(ulong[,] array)
            => new NDArray(array);

        public static implicit operator NDArray(int[,,] array)
            => new NDArray(array);

        public static implicit operator NDArray(byte[,,] array)
            => new NDArray(array);

        public static implicit operator NDArray(float[,,] array)
            => new NDArray(array);

        public static implicit operator NDArray(double[,,] array)
            => new NDArray(array);

        public static implicit operator NDArray(long[,,] array)
            => new NDArray(array);

        public static implicit operator NDArray(bool[,,] array)
            => new NDArray(array);

        public static implicit operator NDArray(uint[,,] array)
            => new NDArray(array);

        public static implicit operator NDArray(ulong[,,] array)
            => new NDArray(array);

        public unsafe static implicit operator bool(NDArray nd)
            => nd.dtype == TF_DataType.TF_BOOL ? *(bool*)nd.data : NDArrayConverter.Scalar<bool>(nd);

        public unsafe static implicit operator byte(NDArray nd)
            => nd.dtype == TF_DataType.TF_UINT8 ? *(byte*)nd.data : NDArrayConverter.Scalar<byte>(nd);

        public unsafe static implicit operator int(NDArray nd)
            => nd.dtype == TF_DataType.TF_INT32 ? *(int*)nd.data : NDArrayConverter.Scalar<int>(nd);

        public unsafe static implicit operator long(NDArray nd)
            => nd.dtype == TF_DataType.TF_INT64 ? *(long*)nd.data : NDArrayConverter.Scalar<long>(nd);

        public unsafe static implicit operator float(NDArray nd)
            => nd.dtype == TF_DataType.TF_FLOAT ? *(float*)nd.data : NDArrayConverter.Scalar<float>(nd);

        public unsafe static implicit operator double(NDArray nd)
            => nd.dtype == TF_DataType.TF_DOUBLE ? *(double*)nd.data : NDArrayConverter.Scalar<double>(nd);

        public static implicit operator NDArray(bool value)
            => new NDArray(value);

        public static implicit operator NDArray(byte value)
            => new NDArray(value);

        public static implicit operator NDArray(int value)
            => new NDArray(value);

        public static implicit operator NDArray(long value)
            => new NDArray(value);

        public static implicit operator NDArray(float value)
            => new NDArray(value);

        public static implicit operator NDArray(double value)
            => new NDArray(value);
    }
}
