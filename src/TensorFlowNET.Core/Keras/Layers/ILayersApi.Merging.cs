using System;
using Tensorflow.NumPy;

namespace Tensorflow.Keras.Layers
{
    public partial interface ILayersApi
    {
        public ILayer Concatenate(int axis = -1);
    }
}
