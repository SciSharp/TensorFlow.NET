using Tensorflow.NumPy;
using System.Collections.Generic;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class TensorFlowOpLayerArgs : LayerArgs
    {
        public NodeDef NodeDef { get; set; }
        public Dictionary<int, NDArray> Constants { get; set; }
    }
}
