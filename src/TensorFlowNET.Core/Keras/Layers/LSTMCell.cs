using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;

namespace Tensorflow.Keras.Layers
{
    public class LSTMCell : Layer
    {
        LSTMCellArgs args;

        public LSTMCell(LSTMCellArgs args)
            : base(args)
        {
            this.args = args;
        }
    }
}
