using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class AutoSerializeLayerArgs: LayerArgs
    {
        [JsonProperty("name")]
        public override string Name { get => base.Name; set => base.Name = value; }
        [JsonProperty("dtype")]
        public override TF_DataType DType { get => base.DType; set => base.DType = value; }
        [JsonProperty("batch_input_shape", NullValueHandling = NullValueHandling.Ignore)]
        public override Shape BatchInputShape { get => base.BatchInputShape; set => base.BatchInputShape = value; }
        [JsonProperty("trainable")]
        public override bool Trainable { get => base.Trainable; set => base.Trainable = value; }
    }
}
