using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition.Rnn;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Saving;

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

        public override void build(KerasShapesWrapper input_shape)
        {
            var single_shape = input_shape.ToSingleShape();
            var input_dim = single_shape[-1];

            kernel = add_weight("kernel", (single_shape[-1], args.Units),
                initializer: args.KernelInitializer
            );

            recurrent_kernel = add_weight("recurrent_kernel", (args.Units, args.Units),
                initializer: args.RecurrentInitializer
            );

            if (args.UseBias)
            {
                bias = add_weight("bias", (args.Units),
                    initializer: args.BiasInitializer
                );
            }

            built = true;
        }

        protected override Tensors Call(Tensors inputs, Tensor mask = null, bool? training = null, Tensors initial_state = null, Tensors constants = null)
        {
            return base.Call(inputs, initial_state, training);
        }
    }
}
