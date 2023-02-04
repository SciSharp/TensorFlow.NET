using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.ArgsDefinition {
    public class SoftmaxArgs : AutoSerializeLayerArgs
    {
        [JsonProperty("axis")]
        public Axis axis { get; set; } = -1;
    }
}
