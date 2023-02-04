using Newtonsoft.Json;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class FlattenArgs : AutoSerializeLayerArgs
    {
        [JsonProperty("data_format")]
        public string DataFormat { get; set; }
    }
}
