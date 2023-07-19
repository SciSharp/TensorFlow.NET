using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.NumPy;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class BidirectionalArgs : AutoSerializeLayerArgs
    {
        [JsonProperty("layer")]
        public ILayer Layer { get; set; }
        [JsonProperty("merge_mode")]
        public string? MergeMode { get; set; }
        [JsonProperty("backward_layer")]
        public ILayer BackwardLayer { get; set; }
        public NDArray Weights { get; set; }
    }

}
