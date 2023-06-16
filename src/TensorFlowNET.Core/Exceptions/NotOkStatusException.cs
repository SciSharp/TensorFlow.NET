using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Exceptions
{
    public class NotOkStatusException : TensorflowException
    {
        public NotOkStatusException() : base()
        {

        }

        public NotOkStatusException(string message) : base(message)
        {

        }
    }
}
