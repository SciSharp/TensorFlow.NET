using System.Data;
using Tensorflow.Keras.ArgsDefinition.Rnn;
using Tensorflow.Keras.Saving;
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
        }

        public override void build(KerasShapesWrapper input_shape)
        {
            var single_shape = input_shape.ToSingleShape();
            var input_dim = single_shape[-1];
            _buildInputShape = input_shape;

            kernel = add_weight("kernel", (single_shape[-1], args.Units),
                initializer: args.KernelInitializer
                //regularizer = self.kernel_regularizer,
                //constraint = self.kernel_constraint,
                //caching_device = default_caching_device,
            );
        }
    }
}