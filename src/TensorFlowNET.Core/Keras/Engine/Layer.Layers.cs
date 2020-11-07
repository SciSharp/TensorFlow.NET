using System.Collections.Generic;

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
