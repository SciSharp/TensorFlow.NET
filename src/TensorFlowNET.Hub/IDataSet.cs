using System;
using System.Collections.Generic;
using System.Text;
using NumSharp;

namespace Tensorflow.Hub
{
    public interface IDataSet
    {
        NDArray Data { get; }
        NDArray Labels { get; }
    }
}
