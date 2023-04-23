using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Saving
{
    public class KerasMetaData
    {
        [JsonProperty("name")]
        public string Name { get; set; }
        [JsonProperty("class_name")]
        public string ClassName { get; set; }
        [JsonProperty("trainable")]
        public bool Trainable { get; set; }
        [JsonProperty("dtype")]
        public TF_DataType DType { get; set; } = TF_DataType.DtInvalid;
        [JsonProperty("is_graph_network")]
        public bool IsGraphNetwork { get; set; }
        [JsonProperty("shared_object_id")]
        public int SharedObjectId { get; set; }
        [JsonProperty("must_restore_from_config")]
        public bool MustRestoreFromConfig { get; set; }
        [JsonProperty("config")]
        public JObject Config { get; set; }
        [JsonProperty("build_input_shape")]
        public KerasShapesWrapper BuildInputShape { get; set; }
        [JsonProperty("batch_input_shape")]
        public KerasShapesWrapper BatchInputShape { get; set; }
        [JsonProperty("activity_regularizer")]
        public IRegularizer ActivityRegularizer { get; set; }
        [JsonProperty("input_spec")]
        public JToken InputSpec { get; set; }
        [JsonProperty("stateful")]
        public bool? Stateful { get; set; }
        [JsonProperty("model_config")]
        public KerasModelConfig? ModelConfig { get; set; }
        [JsonProperty("sparse")]
        public bool Sparse { get; set; }
        [JsonProperty("ragged")]
        public bool Ragged { get; set; }
    }
}
