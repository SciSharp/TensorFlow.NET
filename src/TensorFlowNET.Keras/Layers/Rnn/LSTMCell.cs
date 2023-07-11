using Newtonsoft.Json;
using Serilog.Core;
using System.Diagnostics;
using Tensorflow.Common.Extensions;
using Tensorflow.Common.Types;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Saving;
using Tensorflow.Keras.Utils;

namespace Tensorflow.Keras.Layers
{
    /// <summary>
    /// Cell class for the LSTM layer.
    /// See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
    /// for details about the usage of RNN API.
    /// This class processes one step within the whole time sequence input, whereas
    /// `tf.keras.layer.LSTM` processes the whole sequence.
    /// </summary>
    public class LSTMCell : DropoutRNNCellMixin
    {
        LSTMCellArgs _args;
        IVariableV1 _kernel;
        IVariableV1 _recurrent_kernel;
        IInitializer _bias_initializer;
        IVariableV1 _bias;
        INestStructure<long> _state_size;
        INestStructure<long> _output_size;
        public override INestStructure<long> StateSize => _state_size;

        public override INestStructure<long> OutputSize => _output_size;

        public override bool SupportOptionalArgs => false;
        public LSTMCell(LSTMCellArgs args)
            : base(args)
        {
            _args = args;
            if (args.Units <= 0)
            {
                throw new ValueError(
                            $"units must be a positive integer, got {args.Units}");
            }
            _args.Dropout = Math.Min(1f, Math.Max(0f, this._args.Dropout));
            _args.RecurrentDropout = Math.Min(1f, Math.Max(0f, this._args.RecurrentDropout));
            if (_args.RecurrentDropout != 0f && _args.Implementation != 1)
            {
                Debug.WriteLine("RNN `implementation=2` is not supported when `recurrent_dropout` is set." +
                    "Using `implementation=1`.");
                _args.Implementation = 1;
            }

            _state_size = new NestList<long>(_args.Units, _args.Units);
            _output_size = new NestNode<long>(_args.Units);
        }

        public override void build(KerasShapesWrapper input_shape)
        {
            base.build(input_shape);
            var single_shape = input_shape.ToSingleShape();
            var input_dim = single_shape[-1];
            _kernel = add_weight("kernel", (input_dim, _args.Units * 4),
                initializer: _args.KernelInitializer
            );

            _recurrent_kernel = add_weight("recurrent_kernel", (_args.Units, _args.Units * 4),
                initializer: _args.RecurrentInitializer
            );

            if (_args.UseBias)
            {
                if (_args.UnitForgetBias)
                {
                    Tensor bias_initializer()
                    {
                        return keras.backend.concatenate(
                            new Tensors(
                            _args.BiasInitializer.Apply(new InitializerArgs(shape: (_args.Units))),
                            tf.ones_initializer.Apply(new InitializerArgs(shape: (_args.Units))),
                            _args.BiasInitializer.Apply(new InitializerArgs(shape: (_args.Units)))), axis: 0);
                    }
                }
                else
                {
                    _bias_initializer = _args.BiasInitializer;
                }
                _bias = add_weight("bias", (_args.Units * 4),
                    initializer: _bias_initializer
                    );
            }
            built = true;
        }
        protected override Tensors Call(Tensors inputs, Tensors states = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            var h_tm1 = states[0]; // previous memory state
            var c_tm1 = states[1]; // previous carry state

            var dp_mask = get_dropout_mask_for_cell(inputs, training.Value, count: 4);
            var rec_dp_mask = get_recurrent_dropout_mask_for_cell(
                               h_tm1, training.Value, count: 4);

            Tensor c;
            Tensor o;
            if (_args.Implementation == 1)
            {
                Tensor inputs_i;
                Tensor inputs_f;
                Tensor inputs_c;
                Tensor inputs_o;
                if (0f < _args.Dropout && _args.Dropout < 1f)
                {
                    inputs_i = inputs * dp_mask[0];
                    inputs_f = inputs * dp_mask[1];
                    inputs_c = inputs * dp_mask[2];
                    inputs_o = inputs * dp_mask[3];
                }
                else
                {
                    inputs_i = inputs;
                    inputs_f = inputs;
                    inputs_c = inputs;
                    inputs_o = inputs;
                }
                var k = tf.split(_kernel.AsTensor(), num_split: 4, axis: 1);
                Tensor k_i = k[0], k_f = k[1], k_c = k[2], k_o = k[3];
                var x_i = math_ops.matmul(inputs_i, k_i);
                var x_f = math_ops.matmul(inputs_f, k_f);
                var x_c = math_ops.matmul(inputs_c, k_c);
                var x_o = math_ops.matmul(inputs_o, k_o);
                if (_args.UseBias)
                {
                    var b = tf.split(_bias.AsTensor(), num_split: 4, axis: 0);
                    Tensor b_i = b[0], b_f = b[1], b_c = b[2], b_o = b[3];
                    x_i = gen_nn_ops.bias_add(x_i, b_i);
                    x_f = gen_nn_ops.bias_add(x_f, b_f);
                    x_c = gen_nn_ops.bias_add(x_c, b_c);
                    x_o = gen_nn_ops.bias_add(x_o, b_o);
                }

                Tensor h_tm1_i;
                Tensor h_tm1_f;
                Tensor h_tm1_c;
                Tensor h_tm1_o;
                if (0f < _args.RecurrentDropout && _args.RecurrentDropout < 1f)
                {
                    h_tm1_i = h_tm1 * rec_dp_mask[0];
                    h_tm1_f = h_tm1 * rec_dp_mask[1];
                    h_tm1_c = h_tm1 * rec_dp_mask[2];
                    h_tm1_o = h_tm1 * rec_dp_mask[3];
                }
                else
                {
                    h_tm1_i = h_tm1;
                    h_tm1_f = h_tm1;
                    h_tm1_c = h_tm1;
                    h_tm1_o = h_tm1;
                }
                var x = new Tensor[] { x_i, x_f, x_c, x_o };
                var h_tm1_array = new Tensor[] { h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o };
                (c, o) = _compute_carry_and_output(x, h_tm1_array, c_tm1);
            }
            else
            {
                if (0f < _args.Dropout && _args.Dropout < 1f)
                    inputs = inputs * dp_mask[0];
                var z = math_ops.matmul(inputs, _kernel.AsTensor());
                z += math_ops.matmul(h_tm1, _recurrent_kernel.AsTensor());
                if (_args.UseBias)
                {
                    z = tf.nn.bias_add(z, _bias);
                }
                var z_array = tf.split(z, num_split: 4, axis: 1);
                (c, o) = _compute_carry_and_output_fused(z_array, c_tm1);
            }
            var h = o * _args.Activation.Apply(c);
            // 这里是因为 Tensors 类初始化的时候会把第一个元素之后的元素打包成一个数组
            return new Nest<Tensor>(new INestStructure<Tensor>[] { new NestNode<Tensor>(h), new NestList<Tensor>(h, c) }).ToTensors();
        }

        /// <summary>
        /// Computes carry and output using split kernels.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="h_tm1"></param>
        /// <param name="c_tm1"></param>
        /// <returns></returns>
        /// <exception cref="NotImplementedException"></exception>
        public Tensors _compute_carry_and_output(Tensor[] x, Tensor[] h_tm1, Tensor c_tm1) 
        {
            Tensor x_i = x[0], x_f = x[1], x_c = x[2], x_o = x[3];
            Tensor h_tm1_i = h_tm1[0], h_tm1_f = h_tm1[1], h_tm1_c = h_tm1[2], 
                h_tm1_o = h_tm1[3];

            var _recurrent_kernel_tensor = _recurrent_kernel.AsTensor();
            int startIndex = (int)_recurrent_kernel_tensor.shape[0];
            var _recurrent_kernel_slice = tf.slice(_recurrent_kernel_tensor, 
                new[] { 0, 0 }, new[] { startIndex, _args.Units });
            var i = _args.RecurrentActivation.Apply(
                    x_i + math_ops.matmul(h_tm1_i, _recurrent_kernel_slice));
            _recurrent_kernel_slice = tf.slice(_recurrent_kernel_tensor,
                new[] { 0, _args.Units }, new[] { startIndex, _args.Units});
            var f = _args.RecurrentActivation.Apply(
                    x_f + math_ops.matmul(h_tm1_f, _recurrent_kernel_slice));
            _recurrent_kernel_slice = tf.slice(_recurrent_kernel_tensor,
                new[] { 0, _args.Units * 2 }, new[] { startIndex, _args.Units });
            var c = f * c_tm1 + i * _args.Activation.Apply(
                    x_c + math_ops.matmul(h_tm1_c, _recurrent_kernel_slice));
            _recurrent_kernel_slice = tf.slice(_recurrent_kernel_tensor,
                new[] { 0, _args.Units * 3 }, new[] { startIndex, _args.Units });
            var o = _args.Activation.Apply(
                x_o + math_ops.matmul(h_tm1_o, _recurrent_kernel_slice));

            return new Tensors(c, o);
        }

        /// <summary>
        /// Computes carry and output using fused kernels.
        /// </summary>
        /// <param name="z"></param>
        /// <param name="c_tm1"></param>
        /// <returns></returns>
        public Tensors _compute_carry_and_output_fused(Tensor[] z, Tensor c_tm1)
        {
            Tensor z0 = z[0], z1 = z[1], z2 = z[2], z3 = z[3];
            var i = _args.RecurrentActivation.Apply(z0);
            var f = _args.RecurrentActivation.Apply(z1);
            var c = f * c_tm1 + i * _args.Activation.Apply(z2);
            var o = _args.RecurrentActivation.Apply(z3);
            return new Tensors(c, o);
        }
    }

    
}
