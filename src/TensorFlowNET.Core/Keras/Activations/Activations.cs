using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow.Keras
{
    public delegate Tensor Activation(Tensor features, string name = null);
}
