using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Saving
{
    public class KerasModelConfig
    {
        [JsonProperty("class_name")]
        public string ClassName { get; set; }
        [JsonProperty("config")]
        public JObject Config { get; set; }
    }
}
