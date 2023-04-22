using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using static Google.Protobuf.Reflection.FieldDescriptorProto.Types;

namespace Tensorflow.Keras.Saving
{
    public class FunctionalConfig : IKerasConfig
    {
        [JsonProperty("name")]
        public string Name { get; set; }
        [JsonProperty("layers")]
        public List<LayerConfig> Layers { get; set; }
        [JsonProperty("input_layers")]
        public List<NodeConfig> InputLayers { get; set; }
        [JsonProperty("output_layers")]
        public List<NodeConfig> OutputLayers { get; set; }

        public override string ToString()
            => $"{Name}, {Layers.Count} Layers, {InputLayers.Count} Input Layers, {OutputLayers.Count} Output Layers";
    }
}
