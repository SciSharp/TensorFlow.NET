using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class TensorFlowOpLayerArgs : LayerArgs
    {
        public NodeDef NodeDef { get; set; }
        public Dictionary<int, NDArray> Constants { get; set; }
    }
}
