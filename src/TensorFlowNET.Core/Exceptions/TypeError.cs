using System;

namespace Tensorflow
{
    public class TypeError : Exception
    {
        public TypeError() : base()
        {

        }

        public TypeError(string message) : base(message)
        {

        }
    }
}
