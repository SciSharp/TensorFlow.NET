using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition.Rnn;
using Tensorflow.Keras.Engine;

namespace Tensorflow.Keras.Layers.Rnn
{
    public class SimpleRNNCell : Layer
    {
        public SimpleRNNCell(SimpleRNNArgs args) : base(args)
        {

        }

        protected override void build(Tensors inputs)
        {
            
        }
    }
}
