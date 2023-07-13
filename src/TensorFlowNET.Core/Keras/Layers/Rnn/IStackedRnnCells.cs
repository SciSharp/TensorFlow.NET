using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Layers
{
    public interface IStackedRnnCells : IRnnCell
    {
        int Count { get; }
        IRnnCell this[int idx] { get; }
    }
}
