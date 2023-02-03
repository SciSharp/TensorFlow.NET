using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;

namespace Tensorflow.Keras.Saving
{
    public class LayerConfig: IKerasConfig
    {
        [JsonProperty("name")]
        public string Name { get; set; }
        [JsonProperty("class_name")]
        public string ClassName { get; set; }
        [JsonProperty("config")]
        public LayerArgs Config { get; set; }
        [JsonProperty("inbound_nodes")]
        public List<NodeConfig> InboundNodes { get; set; }
    }
}
