using Newtonsoft.Json;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class UpSampling1DArgs : AutoSerializeLayerArgs
    {
        [JsonProperty("size")]
        public int Size { get; set; }
    }
}
