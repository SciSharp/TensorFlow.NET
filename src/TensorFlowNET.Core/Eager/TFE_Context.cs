using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Eager
{
    public struct TFE_Context
    {
        IntPtr _handle;

        public TFE_Context(IntPtr handle)
            => _handle = handle;

        public static implicit operator TFE_Context(IntPtr handle)
            => new TFE_Context(handle);

        public static implicit operator IntPtr(TFE_Context tensor)
            => tensor._handle;

        public override string ToString()
            => $"TFE_Context {_handle}";
    }
}
