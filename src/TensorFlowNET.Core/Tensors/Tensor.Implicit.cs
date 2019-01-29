using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public partial class Tensor
    {
        public static implicit operator Tensor(double scalar)
        {
            return constant_op.constant(scalar);
        }

        public static implicit operator Tensor(int scalar)
        {
            return constant_op.constant(scalar);
        }

        public static implicit operator int(Tensor tensor)
        {
            return tensor.Data<int>()[0];
        }

        public static implicit operator IntPtr(Tensor tensor)
        {
            return tensor._handle;
        }

        public static implicit operator Tensor(IntPtr handle)
        {
            return new Tensor(handle);
        }
    }
}
