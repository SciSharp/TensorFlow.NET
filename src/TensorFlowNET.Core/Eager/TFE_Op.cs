using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Eager
{
    public struct TFE_Op
    {
        IntPtr _handle;

        public TFE_Op(IntPtr handle)
            => _handle = handle;

        public static implicit operator TFE_Op(IntPtr handle)
            => new TFE_Op(handle);

        public static implicit operator IntPtr(TFE_Op tensor)
            => tensor._handle;

        public override string ToString()
            => $"TFE_Op {_handle}";
    }
}
