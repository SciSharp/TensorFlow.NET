using Tensorflow.Keras.ArgsDefinition.Rnn;
using Tensorflow.Keras.Engine;

namespace Tensorflow.Keras.Layers.Rnn
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
