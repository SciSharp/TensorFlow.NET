using System;
using System.Collections.Generic;
using System.Dynamic;
using System.Linq;
using System.Text;
using Tensorflow.Util;

namespace Tensorflow.Gradients
{
    /// <summary>
    /// Map from tensor_id to internally-defined operation-id of the operation which
    /// produced this tensor. A value of -1 means that the tensor was directly
    /// watched and not the result of any operation in the tape.
    /// </summary>
    public class TensorTape : UnorderedMap<long, long>
    {
        
    }
}
