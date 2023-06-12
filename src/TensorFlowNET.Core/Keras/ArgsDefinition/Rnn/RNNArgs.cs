using Newtonsoft.Json;
using System.Collections.Generic;
using Tensorflow.Keras.Layers.Rnn;

namespace Tensorflow.Keras.ArgsDefinition.Rnn
{
    // TODO(Rinne): add regularizers.
    public class RNNArgs : AutoSerializeLayerArgs
    {
        [JsonProperty("cell")]
        // TODO: the cell should be serialized with `serialize_keras_object`.
        public IRnnCell Cell { get; set; } = null;
        [JsonProperty("cells")]
        public IList<IRnnCell> Cells { get; set; } = null;

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
        // TODO: Add `num_constants` and `zero_output_for_mask`.
        public Dictionary<string, object> Kwargs { get; set; } = null;

        public int Units { get; set; }
        public Activation Activation { get; set; }
        public Activation RecurrentActivation { get; set; }
        public bool UseBias { get; set; } = true;
        public IInitializer KernelInitializer { get; set; }
        public IInitializer RecurrentInitializer { get; set; }
        public IInitializer BiasInitializer { get; set; }
        public float Dropout { get; set; } = .0f;
        public bool ZeroOutputForMask { get; set; } = false;
        public float RecurrentDropout { get; set; } = .0f;

        // kernel_regularizer=None,
        // recurrent_regularizer=None,
        // bias_regularizer=None,
        // activity_regularizer=None,
        // kernel_constraint=None,
        // recurrent_constraint=None,
        // bias_constraint=None,
        // dropout=0.,
        // recurrent_dropout=0.,
        // return_sequences=False,
        // return_state=False,
        // go_backwards=False,
        // stateful=False,
        // unroll=False,
        // **kwargs):
    }
}
