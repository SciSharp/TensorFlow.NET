using System;

namespace Tensorflow
{
    public partial class Tensor
    {
        public static explicit operator bool(Tensor tensor)
        {
            EnsureScalar(tensor);
            return tensor.Data<bool>()[0];
        }

        public static explicit operator sbyte(Tensor tensor)
        {
            EnsureScalar(tensor);
            return tensor.Data<sbyte>()[0];
        }

        public static explicit operator byte(Tensor tensor)
        {
            EnsureScalar(tensor);
            return tensor.Data<byte>()[0];
        }

        public static explicit operator ushort(Tensor tensor)
        {
            EnsureScalar(tensor);
            return tensor.Data<ushort>()[0];
        }

        public static explicit operator short(Tensor tensor)
        {
            EnsureScalar(tensor);
            return tensor.Data<short>()[0];
        }

        public static explicit operator int(Tensor tensor)
        {
            EnsureScalar(tensor);
            return tensor.Data<int>()[0];
        }

        public static explicit operator uint(Tensor tensor)
        {
            EnsureScalar(tensor);
            return tensor.Data<uint>()[0];
        }

        public static explicit operator long(Tensor tensor)
        {
            EnsureScalar(tensor);
            return tensor.Data<long>()[0];
        }

        public static explicit operator ulong(Tensor tensor)
        {
            EnsureScalar(tensor);
            return tensor.Data<ulong>()[0];
        }

        public static explicit operator float(Tensor tensor)
        {
            EnsureScalar(tensor);
            return tensor.Data<float>()[0];
        }

        public static explicit operator double(Tensor tensor)
        {
            EnsureScalar(tensor);
            return tensor.Data<double>()[0];
        }

        private static void EnsureScalar(Tensor tensor)
        {
            if (tensor == null)
            {
                throw new ArgumentNullException(nameof(tensor));
            }

            if (tensor.TensorShape.NDim != 0)
            {
                throw new ArgumentException("Tensor must have 0 dimensions in order to convert to scalar");
            }

            if (tensor.TensorShape.Size != 1)
            {
                throw new ArgumentException("Tensor must have size 1 in order to convert to scalar");
            }
        }

    }
}
