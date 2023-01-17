using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.Layers;

namespace Tensorflow.Keras
{
    public interface IKerasApi
    {
        public ILayersApi layers { get; }
        public IInitializersApi initializers { get; }
    }
}
