using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;

namespace Tensorflow.Keras.Layers
{
    /// <summary>
    /// Long Short-Term Memory layer - Hochreiter 1997.
    /// 
    /// See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
    /// for details about the usage of RNN API.
    /// </summary>
    public class LSTM : RNN
    {
        LSTMArgs args;
        InputSpec[] state_spec;

        int units => args.Units;

        public LSTM(LSTMArgs args) :
            base(args)
        {
            this.args = args;
            state_spec = new[] { units, units }
                .Select(dim => new InputSpec(shape: (-1, dim)))
                .ToArray();
        }

        protected override Tensor call(Tensor inputs, bool is_training = false, Tensor state = null)
        {
            return base.call(inputs, is_training, state);
        }
    }
}
