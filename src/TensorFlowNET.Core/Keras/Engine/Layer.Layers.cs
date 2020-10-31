using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Layers;
using Tensorflow.Operations.Activation;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Engine
{
    public partial class Layer
    {
        protected List<Layer> _layers = new List<Layer>();
        public List<Layer> Layers => _layers;

        protected void StackLayers(params Layer[] layers)
        {
            _layers.AddRange(layers);
        }
    }
}
