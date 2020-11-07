using System;

namespace Tensorflow.Eager
{
    public partial class EagerTensor
    {
        [Obsolete("Implicit conversion of EagerTensor to IntPtr is not supported.", error: true)]
        public static implicit operator IntPtr(EagerTensor tensor)
            => throw new NotSupportedException();
    }
}
