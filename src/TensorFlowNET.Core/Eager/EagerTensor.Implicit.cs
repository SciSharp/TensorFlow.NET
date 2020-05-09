using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Eager;

namespace Tensorflow.Eager
{
    public partial class EagerTensor
    {
        public static explicit operator TFE_TensorHandle(EagerTensor tensor)
            => tensor.tfe_tensor_handle;

        public static implicit operator IntPtr(EagerTensor tensor)
            => tensor.EagerTensorHandle;
    }
}
