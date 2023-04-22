using Newtonsoft.Json;
using Newtonsoft.Json.Serialization;
using Tensorflow.Keras.Saving;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class InputLayerArgs : LayerArgs
    {
        [JsonIgnore]
        public Tensor InputTensor { get; set; }
        [JsonProperty("sparse")]
        public virtual bool Sparse { get; set; }
        [JsonProperty("ragged")]
        public bool Ragged { get; set; }
        [JsonProperty("name")]
        public override string Name { get => base.Name; set => base.Name = value; }
        [JsonProperty("dtype")]
        public override TF_DataType DType { get => base.DType; set => base.DType = value; }
        [JsonProperty("batch_input_shape", NullValueHandling = NullValueHandling.Ignore)]
        public override KerasShapesWrapper BatchInputShape { get => base.BatchInputShape; set => base.BatchInputShape = value; }
    }
}
