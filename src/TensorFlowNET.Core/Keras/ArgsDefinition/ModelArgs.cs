using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class ModelArgs : LayerArgs
    {
        public Tensor[] Inputs { get; set; }
        public Tensor[] Outputs { get; set; }
    }
}
