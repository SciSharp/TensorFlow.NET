using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Losses;

namespace Tensorflow.Keras
{
    public interface IKerasApi
    {
        public ILayersApi layers { get; }
        public ILossesApi losses { get; }
        public IInitializersApi initializers { get; }
    }
}
