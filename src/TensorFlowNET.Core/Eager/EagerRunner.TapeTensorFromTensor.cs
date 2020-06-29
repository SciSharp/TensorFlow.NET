using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Gradients;

namespace Tensorflow.Eager
{
    public partial class EagerRunner
    {
        TapeTensor TapeTensorFromTensor(Tensor tensor)
        {
            return new TapeTensor(tensor.Id, tensor.dtype, tensor.shape);
        }
    }
}
