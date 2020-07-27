using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Engine
{
    /// <summary>
    /// A layer is a callable object that takes as input one or more tensors and
    /// that outputs one or more tensors.
    /// </summary>
    public interface ILayer
    {
        Tensor Apply(Tensor inputs, bool is_training = false);
    }
}
