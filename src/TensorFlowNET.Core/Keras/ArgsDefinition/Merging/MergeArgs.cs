using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.ArgsDefinition
{
    // TODO: complete the implementation
    public class MergeArgs : AutoSerializeLayerArgs
    {
        public Tensors Inputs { get; set; }
        [JsonProperty("axis")]
        public int Axis { get; set; }
    }
}
