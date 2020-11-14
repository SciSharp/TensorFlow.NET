using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;

namespace Tensorflow.Keras.Saving
{
    public class LayerConfig
    {
        public string Name { get; set; }
        public string ClassName { get; set; }
        public LayerArgs Config { get; set; }
        public List<INode> InboundNodes { get; set; }
    }
}
