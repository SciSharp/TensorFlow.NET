using System;
using Tensorflow.NumPy;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public partial class Tensor
    {
        public static implicit operator SafeTensorHandle(Tensor tensor)
            => tensor._handle;
        
        public static implicit operator Operation(Tensor tensor)
            => tensor?.op;

        public static implicit operator Tensor(SafeTensorHandle handle)
            => new Tensor(handle);
    }
}
