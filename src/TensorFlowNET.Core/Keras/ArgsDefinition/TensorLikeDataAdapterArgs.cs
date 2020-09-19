using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class TensorLikeDataAdapterArgs
    {
        public Tensor X { get; set; }
        public Tensor Y { get; set; }
        public int BatchSize { get; set; }
        public int Steps { get; set; }
        public int Epochs { get; set; }
        public bool Shuffle { get; set; }
    }
}
