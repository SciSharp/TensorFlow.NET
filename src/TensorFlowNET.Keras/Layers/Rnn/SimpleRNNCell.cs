using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Saving;
using Tensorflow.Common.Types;
using Tensorflow.Common.Extensions;
using Tensorflow.Keras.Utils;
using Tensorflow.Graphs;

namespace Tensorflow.Keras.Layers
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
        INestStructure<long> _state_size;
        INestStructure<long> _output_size;

        public override INestStructure<long> StateSize => _state_size;
        public override INestStructure<long> OutputSize => _output_size;
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
            _state_size = new NestNode<long>(args.Units);
            _output_size = new NestNode<long>(args.Units);
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

        // TODO(Rinne): revise the trining param (with refactoring of the framework)
        protected override Tensors Call(Tensors inputs, Tensors states = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            // TODO(Rinne): check if it will have multiple tensors when not nested.
            Tensors prev_output = Nest.IsNested(states) ? new Tensors(states[0]) : states;
            var dp_mask = get_dropout_mask_for_cell(inputs, training.Value);
            var rec_dp_mask = get_recurrent_dropout_mask_for_cell(prev_output, training.Value);

            Tensor h;
            var ranks = inputs.rank;
            if (dp_mask != null)
            {

                h = math_ops.matmul(math_ops.multiply(inputs.Single, dp_mask.Single), _kernel.AsTensor());
            }
            else
            {
                h = math_ops.matmul(inputs, _kernel.AsTensor());
            }

            if (_bias != null)
            {
                h = tf.nn.bias_add(h, _bias);
            }

            if (rec_dp_mask != null)
            {
                prev_output = math_ops.multiply(prev_output, rec_dp_mask);
            }
            Tensor output = h + math_ops.matmul(prev_output, _recurrent_kernel.AsTensor());

            if (_args.Activation != null)
            {
                output = _args.Activation.Apply(output);
            }
            if (Nest.IsNested(states))
            {
                return new Nest<Tensor>(new List<Nest<Tensor>> { 
                    new Nest<Tensor>(new List<Nest<Tensor>> { new Nest<Tensor>(output) }), new Nest<Tensor>(output) })
                    .ToTensors();
            }
            else
            {
                return new Tensors(output, output);
            }
        }
    }
}
