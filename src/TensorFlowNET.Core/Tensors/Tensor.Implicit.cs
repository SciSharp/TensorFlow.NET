using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public partial class Tensor
    {
        public static implicit operator IntPtr(Tensor tensor)
        {
            if (tensor._handle == IntPtr.Zero)
                Console.WriteLine("tensor is not allocated.");
            return tensor._handle;
        }

        public static implicit operator Operation(Tensor tensor)
        {
            return tensor.op;
        }

        public static implicit operator Tensor(IntPtr handle)
        {
            return new Tensor(handle);
        }
    }
}
