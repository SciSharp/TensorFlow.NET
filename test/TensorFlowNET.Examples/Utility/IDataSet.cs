using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace TensorFlowNET.Examples.Utility
{
    public interface IDataSet
    {
        NDArray data { get; }
        NDArray labels { get; }
    }
}
