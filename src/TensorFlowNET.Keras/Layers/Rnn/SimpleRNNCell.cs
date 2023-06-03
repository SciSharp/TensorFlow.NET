using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition.Rnn;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Saving;
using Tensorflow.Util;

namespace Tensorflow.Keras.Layers.Rnn
{
    public class SimpleRNNCell : Layer
    {
        SimpleRNNArgs args;
        IVariableV1 kernel;
        IVariableV1 recurrent_kernel;
        IVariableV1 bias;
        DropoutRNNCellMixin DRCMixin;
        public SimpleRNNCell(SimpleRNNArgs args) : base(args)
        {
            this.args = args;
            if (args.Units <= 0)
            {
                throw new ValueError(
                            $"units must be a positive integer, got {args.Units}");
            }
            this.args.Dropout = Math.Min(1f, Math.Max(0f, this.args.Dropout));
            this.args.RecurrentDropout = Math.Min(1f, Math.Max(0f, this.args.RecurrentDropout));
            this.args.state_size = this.args.Units;
            this.args.output_size = this.args.Units;

            DRCMixin = new DropoutRNNCellMixin();
            DRCMixin.dropout = this.args.Dropout;
            DRCMixin.recurrent_dropout = this.args.RecurrentDropout;
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
            Tensor states = initial_state[0];
            var prev_output = nest.is_nested(states) ? states[0] : states;
            var dp_mask = DRCMixin.get_dropout_maskcell_for_cell(inputs, training.Value);
            var rec_dp_mask = DRCMixin.get_recurrent_dropout_maskcell_for_cell(prev_output, training.Value);

            Tensor h;
            var ranks = inputs.rank;
            if (dp_mask != null)
            {
                if (ranks > 2)
                {
                    // 因为multiply函数会自动添加第一个维度，所以加上下标0
                    h = tf.linalg.tensordot(math_ops.multiply(inputs, dp_mask)[0], kernel.AsTensor(), new[,] { { ranks - 1 }, { 0 } });
                }
                else
                {
                    h = math_ops.matmul(math_ops.multiply(inputs, dp_mask)[0], kernel.AsTensor());
                }
            }
            else
            {
                if (ranks > 2)
                {
                    h = tf.linalg.tensordot(inputs, kernel.AsTensor(), new[,] { { ranks - 1 }, { 0 } });
                }
                else
                {
                    h = math_ops.matmul(inputs, kernel.AsTensor());
                }
            }

            if (bias != null)
            {
                h = tf.nn.bias_add(h, bias);
            }

            if (rec_dp_mask != null)
            {
                prev_output = math_ops.multiply(prev_output, rec_dp_mask)[0];
            }

            ranks = prev_output.rank;
            Tensor output;
            if (ranks > 2)
            {
                output = h + tf.linalg.tensordot(prev_output[0], recurrent_kernel.AsTensor(), new[,] { { ranks - 1 }, { 0 } });
            }
            else
            {
                output = h + math_ops.matmul(prev_output, recurrent_kernel.AsTensor());
            }
            Console.WriteLine($"shape of output: {output.shape}");

            if (args.Activation != null)
            {
                output = args.Activation.Apply(output);
            }
            if (nest.is_nested(states))
            {
                return (output, new Tensors { output });
            }
            return (output, output);
        }
       

        public Tensor get_initial_state(Tensors inputs, Tensor batch_size, TF_DataType dtype)
        {
            return RNNUtils.generate_zero_filled_state_for_cell(this, inputs, batch_size, dtype);
        }
    }
}
