using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Eager;

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
            => tensor.op;

        public static implicit operator TF_Tensor(Tensor tensor)
            => new TF_Tensor(tensor._handle);

        public static implicit operator Tensor(IntPtr handle)
            => new Tensor(handle);
    }
}
