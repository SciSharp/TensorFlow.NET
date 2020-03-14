using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Eager
{
    public struct TFE_TensorHandle
    {
        IntPtr _handle;

        public TFE_TensorHandle(IntPtr handle)
            => _handle = handle;

        public static implicit operator TFE_TensorHandle(IntPtr handle)
            => new TFE_TensorHandle(handle);

        public static implicit operator IntPtr(TFE_TensorHandle tensor)
            => tensor._handle;

        public override string ToString()
            => $"TFE_TensorHandle {_handle}";
    }
}
