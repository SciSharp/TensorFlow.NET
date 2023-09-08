using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class GRUCellArgs : AutoSerializeLayerArgs
    {
        [JsonProperty("units")]
        public int Units { get; set; }
        // TODO(Rinne): lack of initialized value of Activation. Merging keras
        // into tf.net could resolve it.
        [JsonProperty("activation")]
        public Activation Activation { get; set; }
        [JsonProperty("recurrent_activation")]
        public Activation RecurrentActivation { get; set; }
        [JsonProperty("use_bias")]
        public bool UseBias { get; set; } = true;
        [JsonProperty("dropout")]
        public float Dropout { get; set; } = .0f;
        [JsonProperty("recurrent_dropout")]
        public float RecurrentDropout { get; set; } = .0f;
        [JsonProperty("kernel_initializer")]
        public IInitializer KernelInitializer { get; set; }
        [JsonProperty("recurrent_initializer")]
        public IInitializer RecurrentInitializer { get; set; }
        [JsonProperty("bias_initializer")]
        public IInitializer BiasInitializer { get; set; }
        [JsonProperty("reset_after")]
        public bool ResetAfter { get;set; }
        [JsonProperty("implementation")]
        public int Implementation { get; set; } = 2;



    }

}
