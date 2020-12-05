using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Exceptions
{
    public class InaccessibleTensorError : TensorflowException
    {
        public InaccessibleTensorError(string message) : base(message)
        {

        }
    }
}
