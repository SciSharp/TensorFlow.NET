using System;

namespace Tensorflow
{
    public class KeyError : Exception
    {
        public KeyError() : base()
        {

        }

        public KeyError(string message) : base(message)
        {

        }
    }
}
