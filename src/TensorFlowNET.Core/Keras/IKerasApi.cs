using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Losses;
using Tensorflow.Keras.Metrics;

namespace Tensorflow.Keras
{
    public interface IKerasApi
    {
        public ILayersApi layers { get; }
        public ILossesApi losses { get; }
        public IMetricsApi metrics { get; }
        public IInitializersApi initializers { get; }
    }
}
