using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.ArgsDefinition {
    public class SoftmaxArgs : LayerArgs
    {
        [JsonProperty("axis")]
        public Axis axis { get; set; } = -1;
        [JsonProperty("name")]
        public override string Name { get => base.Name; set => base.Name = value; }
        [JsonProperty("trainable")]
        public override bool Trainable { get => base.Trainable; set => base.Trainable = value; }
        [JsonProperty("dtype")]
        public override TF_DataType DType { get => base.DType; set => base.DType = value; }
    }
}
