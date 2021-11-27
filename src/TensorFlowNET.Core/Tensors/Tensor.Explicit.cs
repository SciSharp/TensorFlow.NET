using System;
using System.Runtime.CompilerServices;

namespace Tensorflow
{
    public partial class Tensor
    {
        public unsafe static explicit operator bool(Tensor tensor)
        {
            EnsureScalar(tensor);
            EnsureDType(tensor, TF_DataType.TF_BOOL);
            return *(bool*)tensor.buffer;
        }

        public unsafe static explicit operator sbyte(Tensor tensor)
        {
            EnsureScalar(tensor);
            EnsureDType(tensor, TF_DataType.TF_INT8);
            return *(sbyte*)tensor.buffer;
        }

        public unsafe static explicit operator byte(Tensor tensor)
        {
            EnsureScalar(tensor);
            EnsureDType(tensor, TF_DataType.TF_UINT8);
            return *(byte*)tensor.buffer;
        }

        public unsafe static explicit operator ushort(Tensor tensor)
        {
            EnsureScalar(tensor);
            EnsureDType(tensor, TF_DataType.TF_UINT16);
            return *(ushort*)tensor.buffer;
        }

        public unsafe static explicit operator short(Tensor tensor)
        {
            EnsureScalar(tensor);
            EnsureDType(tensor, TF_DataType.TF_INT16);
            return *(short*)tensor.buffer;
        }

        public unsafe static explicit operator int(Tensor tensor)
        {
            EnsureScalar(tensor);
            EnsureDType(tensor, TF_DataType.TF_INT32);
            return *(int*)tensor.buffer;
        }

        public unsafe static explicit operator uint(Tensor tensor)
        {
            EnsureScalar(tensor);
            EnsureDType(tensor, TF_DataType.TF_UINT32);
            return *(uint*)tensor.buffer;
        }

        public unsafe static explicit operator long(Tensor tensor)
        {
            EnsureScalar(tensor);
            EnsureDType(tensor, TF_DataType.TF_INT64);
            return *(long*)tensor.buffer;
        }

        public unsafe static explicit operator ulong(Tensor tensor)
        {
            EnsureScalar(tensor);
            EnsureDType(tensor, TF_DataType.TF_UINT64);
            return *(ulong*)tensor.buffer;
        }

        public unsafe static explicit operator float(Tensor tensor)
        {
            EnsureScalar(tensor);
            EnsureDType(tensor, TF_DataType.TF_FLOAT);
            return *(float*)tensor.buffer;
        }

        public unsafe static explicit operator double(Tensor tensor)
        {
            EnsureScalar(tensor);
            EnsureDType(tensor, TF_DataType.TF_DOUBLE);
            return *(double*)tensor.buffer;
        }

        public unsafe static explicit operator string(Tensor tensor)
        {
            EnsureScalar(tensor);
            EnsureDType(tensor, TF_DataType.TF_STRING);
            return new string((char*)tensor.buffer, 0, (int)tensor.size);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void EnsureDType(Tensor tensor, TF_DataType @is)
        {
            if (tensor.dtype != @is)
                throw new InvalidCastException($"Unable to cast scalar tensor {tensor.dtype} to {@is}");
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void EnsureScalar(Tensor tensor)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            if (tensor.shape.ndim != 0)
                throw new ArgumentException("Tensor must have 0 dimensions in order to convert to scalar");

            if (tensor.shape.size != 1)
                throw new ArgumentException("Tensor must have size 1 in order to convert to scalar");
        }

    }
}
