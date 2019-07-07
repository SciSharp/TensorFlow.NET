using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;

namespace TensorFlowNET.Examples.Text
{
    interface ITextModel
    {
        Tensor is_training { get;  }
        Tensor x { get;}
        Tensor y { get; }
    }
}
