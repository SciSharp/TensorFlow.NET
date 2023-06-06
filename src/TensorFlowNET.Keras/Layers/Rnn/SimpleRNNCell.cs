using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition.Rnn;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Saving;
using Tensorflow.Common.Types;

namespace Tensorflow.Keras.Layers.Rnn
{
    /// <summary>
    /// Cell class for SimpleRNN.
    /// See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
    /// for details about the usage of RNN API.
    /// This class processes one step within the whole time sequence input, whereas
    /// `tf.keras.layer.SimpleRNN` processes the whole sequence.
    /// </summary>
    public class SimpleRNNCell : DropoutRNNCellMixin
    {
        SimpleRNNCellArgs _args;
        IVariableV1 _kernel;
        IVariableV1 _recurrent_kernel;
        IVariableV1 _bias;
        GeneralizedTensorShape _state_size;
        GeneralizedTensorShape _output_size;

        public override GeneralizedTensorShape StateSize => _state_size;
        public override GeneralizedTensorShape OutputSize => _output_size;
        public override bool SupportOptionalArgs => false;

        public SimpleRNNCell(SimpleRNNCellArgs args) : base(args)
        {
            this._args = args;
            if (args.Units <= 0)
            {
                throw new ValueError(
                            $"units must be a positive integer, got {args.Units}");
            }
            this._args.Dropout = Math.Min(1f, Math.Max(0f, this._args.Dropout));
            this._args.RecurrentDropout = Math.Min(1f, Math.Max(0f, this._args.RecurrentDropout));
            _state_size = new GeneralizedTensorShape(args.Units);
            _output_size = new GeneralizedTensorShape(args.Units);
        }

        public override void build(KerasShapesWrapper input_shape)
        {
            // TODO(Rinne): add the cache.
            var single_shape = input_shape.ToSingleShape();
            var input_dim = single_shape[-1];

            _kernel = add_weight("kernel", (single_shape[-1], _args.Units),
                initializer: _args.KernelInitializer
            );

            _recurrent_kernel = add_weight("recurrent_kernel", (_args.Units, _args.Units),
                initializer: _args.RecurrentInitializer
            );

            if (_args.UseBias)
            {
                _bias = add_weight("bias", (_args.Units),
                    initializer: _args.BiasInitializer
                );
            }

            built = true;
        }

        public override (Tensor, Tensors) Call(Tensors inputs, Tensors states, bool? training = null)
        {
            // TODO(Rinne): check if it will have multiple tensors when not nested.
            Tensor prev_output = states[0];
            var dp_mask = get_dropout_maskcell_for_cell(inputs, training.Value);
            var rec_dp_mask = get_recurrent_dropout_maskcell_for_cell(prev_output, training.Value);

            Tensor h;
            var ranks = inputs.rank;
            if (dp_mask != null)
            {
                if (ranks > 2)
                {
                    // 因为multiply函数会自动添加第一个维度，所以加上下标0
                    h = tf.linalg.tensordot(math_ops.multiply(inputs, dp_mask)[0], _kernel.AsTensor(), new[,] { { ranks - 1 }, { 0 } });
                }
                else
                {
                    h = math_ops.matmul(math_ops.multiply(inputs, dp_mask)[0], _kernel.AsTensor());
                }
            }
            else
            {
                if (ranks > 2)
                {
                    h = tf.linalg.tensordot(inputs, _kernel.AsTensor(), new[,] { { ranks - 1 }, { 0 } });
                }
                else
                {
                    h = math_ops.matmul(inputs, _kernel.AsTensor());
                }
            }

            if (_bias != null)
            {
                h = tf.nn.bias_add(h, _bias);
            }

            if (rec_dp_mask != null)
            {
                prev_output = math_ops.multiply(prev_output, rec_dp_mask)[0];
            }

            ranks = prev_output.rank;
            Tensor output;
            if (ranks > 2)
            {
                output = h + tf.linalg.tensordot(prev_output[0], _recurrent_kernel.AsTensor(), new[,] { { ranks - 1 }, { 0 } });
            }
            else
            {
                output = h + math_ops.matmul(prev_output, _recurrent_kernel.AsTensor());
            }
            Console.WriteLine($"shape of output: {output.shape}");

            if (_args.Activation != null)
            {
                output = _args.Activation.Apply(output);
            }
            return (output, new Tensors { output });
        }
    }
}
