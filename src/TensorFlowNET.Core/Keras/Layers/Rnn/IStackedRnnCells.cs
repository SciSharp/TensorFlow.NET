using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Layers.Rnn
{
    public interface IStackedRnnCells : IRnnCell
    {
        int Count { get; }
        IRnnCell this[int idx] { get; }
    }
}
