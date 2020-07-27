using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Engine
{
    public class CallContext
    {
        public CallContextManager enter()
        {
            return new CallContextManager();
        }
    }
}
