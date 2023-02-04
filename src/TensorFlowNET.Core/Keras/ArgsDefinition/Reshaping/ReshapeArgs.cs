using Newtonsoft.Json;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class ReshapeArgs : AutoSerializeLayerArgs
    {
        [JsonProperty("target_shape")]
        public Shape TargetShape { get; set; }
        public object[] TargetShapeObjects { get; set; }
    }
}
