using Newtonsoft.Json;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class RescalingArgs : AutoSerializeLayerArgs
    {
        [JsonProperty("scale")]
        public float Scale { get; set; }
        [JsonProperty("offset")]
        public float Offset { get; set; }
    }
}
