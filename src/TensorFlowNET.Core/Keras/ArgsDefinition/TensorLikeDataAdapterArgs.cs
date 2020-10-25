using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.Engine;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class TensorLikeDataAdapterArgs
    {
        public Tensor X { get; set; }
        public Tensor Y { get; set; }
        public int BatchSize { get; set; } = 32;
        public int Steps { get; set; }
        public int Epochs { get; set; }
        public bool Shuffle { get; set; }
        public int MaxQueueSize { get; set; }
        public int Worker { get; set; }
        public bool UseMultiprocessing { get; set; }
        public Model Model { get; set; }
    }
}
