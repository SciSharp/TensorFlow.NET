using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Eager
{
    public struct TFE_ContextOptions
    {
        IntPtr _handle;

        public TFE_ContextOptions(IntPtr handle)
            => _handle = handle;

        public static implicit operator TFE_ContextOptions(IntPtr handle)
            => new TFE_ContextOptions(handle);

        public static implicit operator IntPtr(TFE_ContextOptions tensor)
            => tensor._handle;

        public override string ToString()
            => $"TFE_ContextOptions {_handle}";
    }
}
