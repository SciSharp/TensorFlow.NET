using Newtonsoft.Json;
using Tensorflow.NumPy;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class CategoryEncodingArgs : AutoSerializeLayerArgs
    {
        [JsonProperty("num_tokens")]
        public int NumTokens { get; set; }
        [JsonProperty("output_mode")]
        public string OutputMode { get; set; }
        [JsonProperty("sparse")]
        public bool Sparse { get; set; }
        public NDArray CountWeights { get; set; }
    }
}
