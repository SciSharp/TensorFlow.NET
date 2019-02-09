using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    /// <summary>
    /// in order to limit function return value 
    /// is Tensor or Operation
    /// </summary>
    public interface ITensorOrOperation
    {
        string Device { get; }
        Operation op { get; }
    }
}
