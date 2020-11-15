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
        public List<NodeConfig> InputLayers { get; set; }
        public List<NodeConfig> OutputLayers { get; set; }

        public override string ToString()
            => $"{Name}, {Layers.Count} Layers, {InputLayers.Count} Input Layers, {OutputLayers.Count} Output Layers";
    }
}
