using System;
using System.Data;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Datasets;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using Tensorflow.Operations.Activation;

namespace Tensorflow
{
    public class KerasApi
    {
        public KerasDataset datasets { get; } = new KerasDataset();
        public Initializers initializers { get; } = new Initializers();
        public Layers layers { get; } = new Layers();

        public class Layers
        {
            public ILayer Dense(int units,
                IActivation activation = null)
                => new Dense(new DenseArgs
                {
                    Units = units,
                    Activation = activation
                });
        }
    }
}
