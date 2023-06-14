using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition.Rnn;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Saving;
<<<<<<< HEAD
using Tensorflow.Common.Types;
using Tensorflow.Common.Extensions;
using Tensorflow.Keras.Utils;
=======
using Tensorflow.Util;
>>>>>>> master

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
<<<<<<< HEAD
        SimpleRNNCellArgs _args;
        IVariableV1 _kernel;
        IVariableV1 _recurrent_kernel;
        IVariableV1 _bias;
        GeneralizedTensorShape _state_size;
        GeneralizedTensorShape _output_size;

        public override GeneralizedTensorShape StateSize => _state_size;
        public override GeneralizedTensorShape OutputSize => _output_size;
        public override bool IsTFRnnCell => true;
        public override bool SupportOptionalArgs => false;

        public SimpleRNNCell(SimpleRNNCellArgs args) : base(args)
        {
            this._args = args;
=======
        SimpleRNNArgs args;
        IVariableV1 kernel;
        IVariableV1 recurrent_kernel;
        IVariableV1 bias;
        DropoutRNNCellMixin DRCMixin;
        public SimpleRNNCell(SimpleRNNArgs args) : base(args)
        {
            this.args = args;
>>>>>>> master
            if (args.Units <= 0)
            {
                throw new ValueError(
                            $"units must be a positive integer, got {args.Units}");
            }
<<<<<<< HEAD
            this._args.Dropout = Math.Min(1f, Math.Max(0f, this._args.Dropout));
            this._args.RecurrentDropout = Math.Min(1f, Math.Max(0f, this._args.RecurrentDropout));
            _state_size = new GeneralizedTensorShape(args.Units);
            _output_size = new GeneralizedTensorShape(args.Units);
=======
            this.args.Dropout = Math.Min(1f, Math.Max(0f, this.args.Dropout));
            this.args.RecurrentDropout = Math.Min(1f, Math.Max(0f, this.args.RecurrentDropout));
            this.args.state_size = this.args.Units;
            this.args.output_size = this.args.Units;

            DRCMixin = new DropoutRNNCellMixin();
            DRCMixin.dropout = this.args.Dropout;
            DRCMixin.recurrent_dropout = this.args.RecurrentDropout;
>>>>>>> master
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

<<<<<<< HEAD
        // TODO(Rinne): revise the trining param (with refactoring of the framework)
        protected override Tensors Call(Tensors inputs, Tensors states = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            // TODO(Rinne): check if it will have multiple tensors when not nested.
            Tensors prev_output = Nest.IsNested(states) ? new Tensors(states[0]) : states;
            var dp_mask = get_dropout_mask_for_cell(inputs, training.Value);
            var rec_dp_mask = get_recurrent_dropout_mask_for_cell(prev_output, training.Value);
=======
        protected override Tensors Call(Tensors inputs, Tensor mask = null, bool? training = null, Tensors initial_state = null, Tensors constants = null)
        {
            Tensor states = initial_state[0];
            var prev_output = nest.is_nested(states) ? states[0] : states;
            var dp_mask = DRCMixin.get_dropout_maskcell_for_cell(inputs, training.Value);
            var rec_dp_mask = DRCMixin.get_recurrent_dropout_maskcell_for_cell(prev_output, training.Value);
>>>>>>> master

            Tensor h;
            var ranks = inputs.rank;
            if (dp_mask != null)
            {
<<<<<<< HEAD

                h = math_ops.matmul(math_ops.multiply(inputs.Single, dp_mask.Single), _kernel.AsTensor());
            }
            else
            {
                h = math_ops.matmul(inputs, _kernel.AsTensor());
            }

            if (_bias != null)
            {
                h = tf.nn.bias_add(h, _bias);
=======
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
>>>>>>> master
            }

            if (rec_dp_mask != null)
            {
<<<<<<< HEAD
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

        public Tensors get_initial_state(Tensors inputs = null, long? batch_size = null, TF_DataType? dtype = null)
        {
            return RnnUtils.generate_zero_filled_state_for_cell(this, inputs, batch_size.Value, dtype.Value);
=======
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
>>>>>>> master
        }
    }
}
