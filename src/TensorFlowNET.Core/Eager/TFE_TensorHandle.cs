using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow.Eager
{
    [StructLayout(LayoutKind.Sequential)]
    public struct TFE_TensorHandle
    {
        IntPtr _handle;

        public static implicit operator IntPtr(TFE_TensorHandle tensor)
            => tensor._handle;

        public override string ToString()
            => $"TFE_TensorHandle 0x{_handle.ToString("x16")}";
    }
}
