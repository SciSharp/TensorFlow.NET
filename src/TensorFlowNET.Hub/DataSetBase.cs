using System;
using System.Collections.Generic;
using System.Text;
using NumSharp;

namespace Tensorflow.Hub
{
    public abstract class DataSetBase : IDataSet
    {
        public NDArray Data { get; protected set; }
        public NDArray Labels { get; protected set; }
    }
}
