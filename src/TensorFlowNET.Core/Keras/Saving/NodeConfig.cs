using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Saving
{
    public class NodeConfig
    {
        public string Name { get; set; }
        public int NodeIndex { get; set; }
        public int TensorIndex { get; set; }
    }
}
