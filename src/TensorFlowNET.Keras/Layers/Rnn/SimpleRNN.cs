using System.Data;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Saving;
using Tensorflow.Operations.Activation;
using static HDF.PInvoke.H5Z;
using static Tensorflow.ApiDef.Types;

namespace Tensorflow.Keras.Layers
{
    public class SimpleRNN : RNN
    {
        SimpleRNNArgs args;
        public SimpleRNN(SimpleRNNArgs args) : base(CreateCellForArgs(args), args)
        {
            this.args = args;
        }

        private static SimpleRNNCell CreateCellForArgs(SimpleRNNArgs args)
        {
            return new SimpleRNNCell(new SimpleRNNCellArgs()
            {
                Units = args.Units,
                Activation = args.Activation,
                UseBias = args.UseBias,
                KernelInitializer = args.KernelInitializer,
                RecurrentInitializer = args.RecurrentInitializer,
                BiasInitializer = args.BiasInitializer,
                Dropout = args.Dropout,
                RecurrentDropout = args.RecurrentDropout,
                DType = args.DType,
                Trainable = args.Trainable,
            });
        }
    }
}