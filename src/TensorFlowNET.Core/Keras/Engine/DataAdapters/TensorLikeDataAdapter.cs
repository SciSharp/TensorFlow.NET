using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Engine.DataAdapters
{
    /// <summary>
    /// Adapter that handles Tensor-like objects, e.g. EagerTensor and NumPy.
    /// </summary>
    public class TensorLikeDataAdapter : IDataAdapter
    {
        public TensorLikeDataAdapter()
        {
            tf.data.Dataset.range(5);
        }

        public bool CanHandle(Tensor x, Tensor y = null)
        {
            throw new NotImplementedException();
        }
    }
}
