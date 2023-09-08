using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Common.Extensions;
using Tensorflow.Common.Types;
using Tensorflow.Keras.Saving;

namespace Tensorflow.Keras.Layers
{
    /// <summary>
    /// Cell class for the GRU layer.
    /// </summary>
    public class GRUCell : DropoutRNNCellMixin
    {
        GRUCellArgs _args;
        IVariableV1 _kernel;
        IVariableV1 _recurrent_kernel;
        IInitializer _bias_initializer;
        IVariableV1 _bias;
        INestStructure<long> _state_size;
        INestStructure<long> _output_size;
        int Units;
        public override INestStructure<long> StateSize => _state_size;

        public override INestStructure<long> OutputSize => _output_size;

        public override bool SupportOptionalArgs => false;
        public GRUCell(GRUCellArgs args) : base(args)
        {
            _args = args;
            if (_args.Units <= 0)
            {
                throw new ValueError(
                            $"units must be a positive integer, got {args.Units}");
            }
            _args.Dropout = Math.Min(1f, Math.Max(0f, _args.Dropout));
            _args.RecurrentDropout = Math.Min(1f, Math.Max(0f, this._args.RecurrentDropout));
            if (_args.RecurrentDropout != 0f && _args.Implementation != 1)
            {
                Debug.WriteLine("RNN `implementation=2` is not supported when `recurrent_dropout` is set." +
                    "Using `implementation=1`.");
                _args.Implementation = 1;
            }
            Units = _args.Units;
            _state_size = new NestList<long>(Units);
            _output_size = new NestNode<long>(Units);
        }

        public override void build(KerasShapesWrapper input_shape)
        {
            //base.build(input_shape);

            var single_shape = input_shape.ToSingleShape();
            var input_dim = single_shape[-1];

            _kernel = add_weight("kernel", (input_dim, _args.Units * 3),
                initializer: _args.KernelInitializer
            );

            _recurrent_kernel = add_weight("recurrent_kernel", (Units, Units * 3),
                initializer: _args.RecurrentInitializer
            );
            if (_args.UseBias)
            {
                Shape bias_shape;
                if (!_args.ResetAfter)
                {
                    bias_shape = new Shape(3 * Units);
                }
                else
                {
                    bias_shape = (2, 3 *  Units);
                }
                _bias = add_weight("bias", bias_shape,
                    initializer: _bias_initializer
                    );
            }
            built = true;
        }

        protected override Tensors Call(Tensors inputs, Tensors states = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            var h_tm1 = states.IsNested() ? states[0] : states.Single();
            var dp_mask = get_dropout_mask_for_cell(inputs, training.Value, count: 3);
            var rec_dp_mask = get_recurrent_dropout_mask_for_cell(h_tm1, training.Value, count: 3);

            IVariableV1 input_bias = _bias;
            IVariableV1 recurrent_bias = _bias;
            if (_args.UseBias)
            {
                if (!_args.ResetAfter)
                {
                    input_bias = _bias;
                    recurrent_bias = null;
                }
                else
                {
                    input_bias = tf.Variable(tf.unstack(_bias.AsTensor())[0]);
                    recurrent_bias = tf.Variable(tf.unstack(_bias.AsTensor())[1]);
                }
            }


            Tensor hh;
            Tensor z;
            if ( _args.Implementation == 1)
            {
                Tensor inputs_z;
                Tensor inputs_r;
                Tensor inputs_h;
                if (0f < _args.Dropout && _args.Dropout < 1f)
                {
                    inputs_z = inputs * dp_mask[0];
                    inputs_r = inputs * dp_mask[1];
                    inputs_h = inputs * dp_mask[2];
                }
                else
                {
                    inputs_z = inputs.Single();
                    inputs_r = inputs.Single();
                    inputs_h = inputs.Single();
                }


                int startIndex = (int)_kernel.AsTensor().shape[0];
                var _kernel_slice = tf.slice(_kernel.AsTensor(),
                    new[] { 0, 0 }, new[] { startIndex, Units });
                var x_z = math_ops.matmul(inputs_z, _kernel_slice);
                _kernel_slice = tf.slice(_kernel.AsTensor(),
                    new[] { 0, Units }, new[] { Units, Units});
                var x_r = math_ops.matmul(
                    inputs_r, _kernel_slice);
                int endIndex = (int)_kernel.AsTensor().shape[1];
                _kernel_slice = tf.slice(_kernel.AsTensor(),
                    new[] { 0, Units * 2 }, new[] { startIndex, endIndex - Units * 2 });
                var x_h = math_ops.matmul(inputs_h, _kernel_slice);

                if(_args.UseBias)
                {
                    x_z = tf.nn.bias_add(
                        x_z, tf.Variable(input_bias.AsTensor()[$":{Units}"]));
                    x_r = tf.nn.bias_add(
                        x_r, tf.Variable(input_bias.AsTensor()[$"{Units}:{Units * 2}"]));
                    x_h = tf.nn.bias_add(
                        x_h, tf.Variable(input_bias.AsTensor()[$"{Units * 2}:"]));
                }

                Tensor h_tm1_z;
                Tensor h_tm1_r;
                Tensor h_tm1_h;
                if (0f < _args.RecurrentDropout && _args.RecurrentDropout < 1f)
                {
                    h_tm1_z = h_tm1 * rec_dp_mask[0];
                    h_tm1_r = h_tm1 * rec_dp_mask[1];
                    h_tm1_h = h_tm1 * rec_dp_mask[2];
                }
                else
                {
                    h_tm1_z = h_tm1;
                    h_tm1_r = h_tm1;
                    h_tm1_h = h_tm1;
                }

                startIndex = (int)_recurrent_kernel.AsTensor().shape[0];
                var _recurrent_kernel_slice = tf.slice(_recurrent_kernel.AsTensor(),
                    new[] { 0, 0 }, new[] { startIndex, Units });
                var recurrent_z = math_ops.matmul(
                    h_tm1_z, _recurrent_kernel_slice);
                _recurrent_kernel_slice = tf.slice(_recurrent_kernel.AsTensor(),
                    new[] { 0, Units }, new[] { startIndex, Units});
                var recurrent_r = math_ops.matmul(
                    h_tm1_r, _recurrent_kernel_slice);
                if(_args.ResetAfter && _args.UseBias)
                {
                    recurrent_z = tf.nn.bias_add(
                        recurrent_z, tf.Variable(recurrent_bias.AsTensor()[$":{Units}"]));
                    recurrent_r = tf.nn.bias_add(
                        recurrent_r, tf.Variable(recurrent_bias.AsTensor()[$"{Units}: {Units * 2}"]));
                }
                z = _args.RecurrentActivation.Apply(x_z + recurrent_z);
                var r = _args.RecurrentActivation.Apply(x_r + recurrent_r);

                Tensor recurrent_h;
                if (_args.ResetAfter)
                {
                    endIndex = (int)_recurrent_kernel.AsTensor().shape[1];
                    _recurrent_kernel_slice = tf.slice(_recurrent_kernel.AsTensor(),
                        new[] { 0, Units * 2 }, new[] { startIndex, endIndex - Units * 2 });
                    recurrent_h = math_ops.matmul(
                        h_tm1_h, _recurrent_kernel_slice);
                    if(_args.UseBias)
                    {
                        recurrent_h = tf.nn.bias_add(
                            recurrent_h, tf.Variable(recurrent_bias.AsTensor()[$"{Units * 2}:"]));
                    }
                    recurrent_h *= r;
                }
                else
                {
                    _recurrent_kernel_slice = tf.slice(_recurrent_kernel.AsTensor(),
                        new[] { 0, Units * 2 }, new[] { startIndex, endIndex - Units * 2 });
                    recurrent_h = math_ops.matmul(
                        r * h_tm1_h, _recurrent_kernel_slice);
                }
                hh = _args.Activation.Apply(x_h + recurrent_h);
            }
            else
            {
                if (0f < _args.Dropout && _args.Dropout < 1f)
                {
                    inputs = inputs * dp_mask[0];
                }

                var matrix_x = math_ops.matmul(inputs, _kernel.AsTensor());
                if(_args.UseBias)
                {
                    matrix_x = tf.nn.bias_add(matrix_x, input_bias);
                }
                var matrix_x_spilted = tf.split(matrix_x, 3, axis: -1);
                var x_z = matrix_x_spilted[0];
                var x_r = matrix_x_spilted[1];
                var x_h = matrix_x_spilted[2];

                Tensor matrix_inner;
                if (_args.ResetAfter)
                {
                    matrix_inner = math_ops.matmul(h_tm1, _recurrent_kernel.AsTensor());
                    if ( _args.UseBias)
                    {
                        matrix_inner = tf.nn.bias_add(
                            matrix_inner, recurrent_bias);
                    }
                }
                else
                {
                    var startIndex = (int)_recurrent_kernel.AsTensor().shape[0];
                    var _recurrent_kernel_slice = tf.slice(_recurrent_kernel.AsTensor(),
                        new[] { 0, 0 }, new[] { startIndex, Units * 2 });
                    matrix_inner = math_ops.matmul(
                        h_tm1, _recurrent_kernel_slice);
                }

                var matrix_inner_splitted = tf.split(matrix_inner, new int[] {Units, Units, -1}, axis:-1);
                var recurrent_z = matrix_inner_splitted[0];
                var recurrent_r = matrix_inner_splitted[0];
                var recurrent_h = matrix_inner_splitted[0];

                z = _args.RecurrentActivation.Apply(x_z + recurrent_z);
                var r = _args.RecurrentActivation.Apply(x_r + recurrent_r);

                if(_args.ResetAfter)
                {
                    recurrent_h = r * recurrent_h;
                }
                else
                {
                    var startIndex = (int)_recurrent_kernel.AsTensor().shape[0];
                    var endIndex = (int)_recurrent_kernel.AsTensor().shape[1];
                    var _recurrent_kernel_slice = tf.slice(_recurrent_kernel.AsTensor(),
                        new[] { 0, 2*Units }, new[] { startIndex, endIndex - 2 * Units });
                    recurrent_h = math_ops.matmul(
                        r * h_tm1, _recurrent_kernel_slice);
                }
                hh = _args.Activation.Apply(x_h + recurrent_h);
            }
            var h = z * h_tm1 + (1 - z) * hh;
            if (states.IsNested())
            {
                var new_state = new NestList<Tensor>(h);
                return new Nest<Tensor>(new INestStructure<Tensor>[] { new NestNode<Tensor>(h), new_state }).ToTensors();
            }
            else
            {
                return new Nest<Tensor>(new INestStructure<Tensor>[] { new NestNode<Tensor>(h), new NestNode<Tensor>(h)}).ToTensors();
            }

        }
    }
}
