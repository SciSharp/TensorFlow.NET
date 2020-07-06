using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Eager;

namespace Tensorflow.Eager
{
    public partial class EagerTensor
    {
        [Obsolete("Implicit conversion of EagerTensor to IntPtr is not supported.", error: true)]
        public static implicit operator IntPtr(EagerTensor tensor)
            => throw new NotSupportedException();
    }
}
