using System.Linq;
using Tensorflow.Keras.ArgsDefinition.Rnn;
using Tensorflow.Keras.Engine;

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

        protected override Tensors Call(Tensors inputs, Tensor mask = null, bool? training = null, Tensors initial_state = null, Tensors constants = null)
        {
            return base.Call(inputs, initial_state: initial_state, training: training);
        }
    }
}
