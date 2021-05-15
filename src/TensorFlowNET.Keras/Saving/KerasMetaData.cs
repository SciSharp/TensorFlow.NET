using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Saving
{
    public class KerasMetaData
    {
        public string Name { get; set; }
        [JsonProperty("class_name")]
        public string ClassName { get; set; }
        [JsonProperty("is_graph_network")]
        public bool IsGraphNetwork { get; set; }
        [JsonProperty("shared_object_id")]
        public int SharedObjectId { get; set; }
        [JsonProperty("must_restore_from_config")]
        public bool MustRestoreFromConfig { get; set; }
        public ModelConfig Config { get; set; }
        [JsonProperty("build_input_shape")]
        public TensorShapeConfig BuildInputShape { get; set; }
    }
}
