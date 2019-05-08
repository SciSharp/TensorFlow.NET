using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;

namespace TensorFlowNET.Examples.Text.cnn_models
{
    interface ITextClassificationModel
    {
        Tensor is_training { get;  }
        Tensor x { get;}
        Tensor y { get; }
    }
}
