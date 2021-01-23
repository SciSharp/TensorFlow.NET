using System;

namespace Tensorflow.Eager
{
    public partial class EagerTensor
    {
        public static implicit operator IntPtr(EagerTensor tensor)
            => tensor.EagerTensorHandle.DangerousGetHandle();
    }
}
