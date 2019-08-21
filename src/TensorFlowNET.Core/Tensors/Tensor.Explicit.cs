using System;
using System.Runtime.CompilerServices;

namespace Tensorflow
{
    public partial class Tensor
    {
        public static explicit operator bool(Tensor tensor)
        {
            unsafe
            {
                EnsureScalar(tensor);
                return *(bool*) tensor.buffer;
            }
        }

        public static explicit operator sbyte(Tensor tensor)
        {
            unsafe
            {
                EnsureScalar(tensor);
                return *(sbyte*) tensor.buffer;
            }
        }

        public static explicit operator byte(Tensor tensor)
        {
            unsafe
            {
                EnsureScalar(tensor);
                return *(byte*) tensor.buffer;
            }
        }

        public static explicit operator ushort(Tensor tensor)
        {
            unsafe
            {
                EnsureScalar(tensor);
                return *(ushort*) tensor.buffer;
            }
        }

        public static explicit operator short(Tensor tensor)
        {
            unsafe
            {
                EnsureScalar(tensor);
                return *(short*) tensor.buffer;
            }
        }

        public static explicit operator int(Tensor tensor)
        {
            unsafe
            {
                EnsureScalar(tensor);
                return *(int*) tensor.buffer;
            }
        }

        public static explicit operator uint(Tensor tensor)
        {
            unsafe
            {
                EnsureScalar(tensor);
                return *(uint*) tensor.buffer;
            }
        }

        public static explicit operator long(Tensor tensor)
        {
            unsafe
            {
                EnsureScalar(tensor);
                return *(long*) tensor.buffer;
            }
        }

        public static explicit operator ulong(Tensor tensor)
        {
            unsafe
            {
                EnsureScalar(tensor);
                return *(ulong*) tensor.buffer;
            }
        }

        public static explicit operator float(Tensor tensor)
        {
            unsafe
            {
                EnsureScalar(tensor);
                return *(float*) tensor.buffer;
            }
        }

        public static explicit operator double(Tensor tensor)
        {
            unsafe
            {
                EnsureScalar(tensor);
                return *(double*) tensor.buffer;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void EnsureScalar(Tensor tensor)
        {
            if (tensor == null)
            {
                throw new ArgumentNullException(nameof(tensor));
            }

            if (tensor.TensorShape.ndim != 0)
            {
                throw new ArgumentException("Tensor must have 0 dimensions in order to convert to scalar");
            }

            if (tensor.TensorShape.size != 1)
            {
                throw new ArgumentException("Tensor must have size 1 in order to convert to scalar");
            }
        }

    }
}
