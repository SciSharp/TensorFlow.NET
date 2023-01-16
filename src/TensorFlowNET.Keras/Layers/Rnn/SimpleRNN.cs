using System.Data;
using Tensorflow.Keras.ArgsDefinition.Rnn;
using Tensorflow.Operations.Activation;
using static HDF.PInvoke.H5Z;
using static Tensorflow.ApiDef.Types;

namespace Tensorflow.Keras.Layers.Rnn
{
    public class SimpleRNN : RNN
    {
        SimpleRNNArgs args;
        public SimpleRNN(SimpleRNNArgs args) : base(args)
        {
            this.args = args;
            cell = new SimpleRNNCell(args);
        }
    }
}