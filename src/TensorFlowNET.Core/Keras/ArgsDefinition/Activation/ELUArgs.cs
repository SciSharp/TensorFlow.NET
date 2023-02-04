using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.ArgsDefinition {
    public class ELUArgs : AutoSerializeLayerArgs
    {
        [JsonProperty("alpha")]
        public float Alpha { get; set; } = 0.1f;
    }
}
