using System;

namespace Tensorflow
{
    public struct TF_Tensor
    {
        IntPtr _handle;

        public TF_Tensor(IntPtr handle)
            => _handle = handle;

        public static implicit operator TF_Tensor(IntPtr handle)
            => new TF_Tensor(handle);

        public static implicit operator IntPtr(TF_Tensor tensor)
            => tensor._handle;

        public override string ToString()
            => $"TF_Tensor 0x{_handle.ToString("x16")}";
    }
}
