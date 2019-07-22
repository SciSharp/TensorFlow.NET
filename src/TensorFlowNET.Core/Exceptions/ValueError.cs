using System;

namespace Tensorflow
{
    public class ValueError : Exception
    {
        public ValueError() : base()
        {

        }

        public ValueError(string message) : base(message)
        {

        }
    }
}
