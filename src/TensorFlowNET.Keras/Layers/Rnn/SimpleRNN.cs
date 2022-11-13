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
        SimpleRNNCell cell;
        public SimpleRNN(SimpleRNNArgs args) : base(args)
        {
            this.args = args;
        }

        protected override void build(Tensors inputs)
        {
            var input_shape = inputs.shape;
            var input_dim = input_shape[-1];

            kernel = add_weight("kernel", (input_shape[-1], args.Units),
                initializer: args.KernelInitializer
                //regularizer = self.kernel_regularizer,
                //constraint = self.kernel_constraint,
                //caching_device = default_caching_device,
            );
        }
    }
}