using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition.Rnn;
using Tensorflow.Keras.Engine;

namespace Tensorflow.Keras.Layers.Rnn
{
    public class SimpleRNNCell : Layer
    {
        SimpleRNNArgs args;
        IVariableV1 kernel;
        IVariableV1 recurrent_kernel;
        IVariableV1 bias;

        public SimpleRNNCell(SimpleRNNArgs args) : base(args)
        {
            this.args = args;
        }

        public override void build(Shape input_shape)
        {
            var input_dim = input_shape[-1];

            kernel = add_weight("kernel", (input_shape[-1], args.Units),
                initializer: args.KernelInitializer
            );

            recurrent_kernel = add_weight("recurrent_kernel", (args.Units, args.Units),
                initializer: args.RecurrentInitializer
            );

            if (args.UseBias)
            {
                bias = add_weight("bias", (args.Units),
                    initializer: args.RecurrentInitializer
                );
            }

            built = true;
        }
    }
}
