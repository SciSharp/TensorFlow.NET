using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.Engine;

namespace Tensorflow.Keras.Callbacks
{
    public class CallbackParams
    {
        public IModel Model { get; set; }
        public int Verbose { get; set; }
        public int Epochs { get; set; }
        public long Steps { get; set; }
    }
}
