using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Eager
{
    public struct TFE_Executor
    {
        IntPtr _handle;

        public TFE_Executor(IntPtr handle)
            => _handle = handle;

        public static implicit operator TFE_Executor(IntPtr handle)
            => new TFE_Executor(handle);

        public static implicit operator IntPtr(TFE_Executor tensor)
            => tensor._handle;

        public override string ToString()
            => $"TFE_Executor {_handle}";
    }
}
