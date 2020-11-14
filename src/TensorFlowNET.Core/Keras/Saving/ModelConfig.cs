using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.Engine;

namespace Tensorflow.Keras.Saving
{
    public class ModelConfig
    {
        public string Name { get; set; }
        public List<LayerConfig> Layers { get; set; }
        public List<ILayer> InputLayers { get; set; }
        public List<ILayer> OutputLayers { get; set; }
    }
}
