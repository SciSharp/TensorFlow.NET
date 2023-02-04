using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.ArgsDefinition
{
    // TODO: complete the implementation
    public class MergeArgs : LayerArgs
    {
        public Tensors Inputs { get; set; }
        public int Axis { get; set; }
    }
}
