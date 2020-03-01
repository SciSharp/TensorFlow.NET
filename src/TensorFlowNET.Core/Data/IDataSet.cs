using System;
using System.Collections.Generic;
using System.Text;
using NumSharp;

namespace Tensorflow
{
    public interface IDataSet
    {
        NDArray Data { get; }
        NDArray Labels { get; }
    }
}
