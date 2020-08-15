using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;

namespace Tensorflow.Keras.Layers
{
    public class RNN : Layer
    {
        public RNN(RNNArgs args)
        : base(args)
        {

        }

        protected Tensor get_initial_state(Tensor inputs)
        {
            return _generate_zero_filled_state_for_cell(null, null);
        }

        Tensor _generate_zero_filled_state_for_cell(LSTMCell cell, Tensor batch_size)
        {
            throw new NotImplementedException("");
        }
    }
}
