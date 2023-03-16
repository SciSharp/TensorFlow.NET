using Newtonsoft.Json;

namespace Tensorflow.Keras.ArgsDefinition;

public class NormalizationArgs : PreprocessingLayerArgs
{
    [JsonProperty("axis")]
    public Axis? Axis { get; set; }
    [JsonProperty("mean")]
    public float? Mean { get; set; }
    [JsonProperty("variance")]
    public float? Variance { get; set; }
    
    public bool Invert { get; set; } = false;
}
