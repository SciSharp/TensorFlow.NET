using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.Saving.Common;

namespace Tensorflow.Keras.Saving
{
    [JsonConverter(typeof(CustomizedNodeConfigJsonConverter))]
    public class NodeConfig : IKerasConfig
    {
        public string Name { get; set; }
        public int NodeIndex { get; set; }
        public int TensorIndex { get; set; }

        public override string ToString()
            => $"{Name}, {NodeIndex}, {TensorIndex}";
    }
}
