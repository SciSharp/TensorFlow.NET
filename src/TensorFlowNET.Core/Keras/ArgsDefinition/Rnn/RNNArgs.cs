using Newtonsoft.Json;
using System.Collections.Generic;
using Tensorflow.Keras.Layers;

namespace Tensorflow.Keras.ArgsDefinition
{
    // TODO(Rinne): add regularizers.
    public class RNNArgs : AutoSerializeLayerArgs
    {
        [JsonProperty("return_sequences")]
        public bool ReturnSequences { get; set; } = false;
        [JsonProperty("return_state")]
        public bool ReturnState { get; set; } = false;
        [JsonProperty("go_backwards")]
        public bool GoBackwards { get; set; } = false;
        [JsonProperty("stateful")]
        public bool Stateful { get; set; } = false;
        [JsonProperty("unroll")]
        public bool Unroll { get; set; } = false;
        [JsonProperty("time_major")]
        public bool TimeMajor { get; set; } = false;

        public int? InputDim { get; set; }
        public int? InputLength { get; set; }
        // TODO: Add `num_constants` and `zero_output_for_mask`.
        [JsonProperty("units")]
        public int Units { get; set; }
        [JsonProperty("activation")]
        public Activation Activation { get; set; }
        [JsonProperty("recurrent_activation")]
        public Activation RecurrentActivation { get; set; }
        [JsonProperty("use_bias")]
        public bool UseBias { get; set; } = true;
        public IInitializer KernelInitializer { get; set; }
        public IInitializer RecurrentInitializer { get; set; }
        public IInitializer BiasInitializer { get; set; }
        [JsonProperty("dropout")]
        public float Dropout { get; set; } = .0f;
        [JsonProperty("zero_output_for_mask")]
        public bool ZeroOutputForMask { get; set; } = false;
        [JsonProperty("recurrent_dropout")]
        public float RecurrentDropout { get; set; } = .0f;

        public RNNArgs Clone()
        {
            return (RNNArgs)MemberwiseClone();
        }
    }
}
