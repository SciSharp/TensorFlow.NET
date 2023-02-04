using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class LeakyReLuArgs : AutoSerializeLayerArgs
    {
        /// <summary>
        /// Negative slope coefficient.
        /// </summary>
        [JsonProperty("alpha")]
        public float Alpha { get; set; } = 0.3f;
    }
}
