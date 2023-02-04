using Newtonsoft.Json;

namespace Tensorflow.Keras.ArgsDefinition {
    public class PermuteArgs : AutoSerializeLayerArgs
    {
        [JsonProperty("dims")]
        public int[] dims { get; set; }
    }
}
