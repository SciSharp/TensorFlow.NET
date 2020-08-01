using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class InputLayerArgs : LayerArgs
    {
        public Tensor InputTensor { get; set; }
        public bool Sparse { get; set; }
        public bool Ragged { get; set; }
    }
}
