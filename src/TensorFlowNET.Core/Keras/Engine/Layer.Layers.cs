using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Layers;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Engine
{
    public partial class Layer
    {
        protected List<Layer> _layers = new List<Layer>();

        protected Layer Dense(int units,
            Activation activation = null,
            TensorShape input_shape = null)
        {
            var layer = new Dense(new DenseArgs
            {
                Units = units,
                Activation = activation ?? tf.keras.activations.Linear,
                InputShape = input_shape
            });

            _layers.Add(layer);
            return layer;
        }
    }
}
