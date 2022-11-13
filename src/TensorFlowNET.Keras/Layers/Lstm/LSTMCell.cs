using Tensorflow.Keras.ArgsDefinition.Lstm;
using Tensorflow.Keras.Engine;

namespace Tensorflow.Keras.Layers.Lstm
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
