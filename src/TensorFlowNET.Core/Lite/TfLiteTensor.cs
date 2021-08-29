using System;

namespace Tensorflow.Lite
{
    public struct TfLiteTensor
    {
        IntPtr _handle;

        public TfLiteTensor(IntPtr handle)
            => _handle = handle;

        public static implicit operator TfLiteTensor(IntPtr handle)
            => new TfLiteTensor(handle);

        public static implicit operator IntPtr(TfLiteTensor tensor)
            => tensor._handle;

        public override string ToString()
            => $"TfLiteTensor 0x{_handle.ToString("x16")}";
    }
}
