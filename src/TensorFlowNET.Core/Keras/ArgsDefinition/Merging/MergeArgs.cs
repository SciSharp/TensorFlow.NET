using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class MergeArgs : LayerArgs
    {
        public Tensors Inputs { get; set; }
        public int Axis { get; set; }
    }
}
