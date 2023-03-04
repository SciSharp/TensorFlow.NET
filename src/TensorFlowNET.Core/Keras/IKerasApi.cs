using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.Engine;
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

        /// <summary>
        /// `Model` groups layers into an object with training and inference features.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="output"></param>
        /// <returns></returns>
        public IModel Model(Tensors inputs, Tensors outputs, string name = null);
    }
}
