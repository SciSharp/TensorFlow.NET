using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Eager
{
    public struct EagerTensorHandle
    {
        IntPtr _handle;

        public EagerTensorHandle(IntPtr handle)
            => _handle = handle;

        public static implicit operator EagerTensorHandle(IntPtr handle)
            => new EagerTensorHandle(handle);

        public static implicit operator IntPtr(EagerTensorHandle tensor)
            => tensor._handle;

        public static implicit operator Tensor(EagerTensorHandle tensor)
            => new EagerTensor(tensor._handle);

        public override string ToString()
            => $"EagerTensorHandle 0x{_handle.ToString("x16")}";
    }
}
