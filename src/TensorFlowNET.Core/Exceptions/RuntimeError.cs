using System;

namespace Tensorflow
{
    public class RuntimeError : Exception
    {
        public RuntimeError() : base()
        {

        }

        public RuntimeError(string message) : base(message)
        {

        }
    }
}
