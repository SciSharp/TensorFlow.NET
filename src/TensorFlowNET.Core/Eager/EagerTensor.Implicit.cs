using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Eager;

namespace Tensorflow.Eager
{
    public partial class EagerTensor
    {
        public static implicit operator IntPtr(EagerTensor tensor)
            => tensor.EagerTensorHandle;
    }
}
