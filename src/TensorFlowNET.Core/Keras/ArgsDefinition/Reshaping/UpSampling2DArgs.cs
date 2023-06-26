using Newtonsoft.Json;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class UpSampling2DArgs : AutoSerializeLayerArgs
    {
        [JsonProperty("size")]
        public Shape Size { get; set; }
        [JsonProperty("data_format")]
        public string DataFormat { get; set; } = "channels_last";
        /// <summary>
        /// 'nearest', 'bilinear'
        /// </summary>
        [JsonProperty("interpolation")]
        public string Interpolation { get; set; } = "nearest";
    }
}
