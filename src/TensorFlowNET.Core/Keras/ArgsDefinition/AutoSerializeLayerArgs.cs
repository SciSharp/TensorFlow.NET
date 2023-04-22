using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.Saving;

namespace Tensorflow.Keras.ArgsDefinition
{
    /// <summary>
    /// This class has nothing but the attributes different from `LayerArgs`.
    /// It's used to serialize the model to `tf` format. 
    /// If the `get_config` of a `Layer` in python code of tensorflow contains `super().get_config`,
    /// then the Arg definition should inherit `AutoSerializeLayerArgs` instead of `LayerArgs`.
    /// </summary>
    public class AutoSerializeLayerArgs: LayerArgs
    {
        [JsonProperty("name")]
        public override string Name { get => base.Name; set => base.Name = value; }
        [JsonProperty("dtype")]
        public override TF_DataType DType { get => base.DType; set => base.DType = value; }
        [JsonProperty("batch_input_shape", NullValueHandling = NullValueHandling.Ignore)]
        public override KerasShapesWrapper BatchInputShape { get => base.BatchInputShape; set => base.BatchInputShape = value; }
        [JsonProperty("trainable")]
        public override bool Trainable { get => base.Trainable; set => base.Trainable = value; }
    }
}
