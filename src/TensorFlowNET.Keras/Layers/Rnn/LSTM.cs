using System.Linq;
using Tensorflow.Keras.ArgsDefinition.Rnn;
using Tensorflow.Keras.Engine;
using Tensorflow.Common.Types;

namespace Tensorflow.Keras.Layers.Rnn
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

        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            return base.Call(inputs, initial_state: state, training: training);
        }
    }
}
