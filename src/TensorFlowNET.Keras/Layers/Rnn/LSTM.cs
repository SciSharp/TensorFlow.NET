using System.Linq;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Common.Types;
using Tensorflow.Common.Extensions;
using Tensorflow.Keras.Saving;

namespace Tensorflow.Keras.Layers
{
    /// <summary>
    /// Long Short-Term Memory layer - Hochreiter 1997.
    /// 
    /// See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
    /// for details about the usage of RNN API.
    /// </summary>
    public class LSTM : RNN
    {
        LSTMArgs _args;
        InputSpec[] _state_spec;
        InputSpec _input_spec;
        bool _could_use_gpu_kernel;
        public LSTMArgs Args { get => _args; }
        public LSTM(LSTMArgs args) :
            base(CreateCell(args), args)
        {
            _args = args;
            _input_spec = new InputSpec(ndim: 3);
            _state_spec = new[] { args.Units, args.Units }.Select(dim => new InputSpec(shape: (-1, dim))).ToArray();
            _could_use_gpu_kernel = args.Activation == keras.activations.Tanh
                && args.RecurrentActivation == keras.activations.Sigmoid
                && args.RecurrentDropout == 0 && !args.Unroll && args.UseBias
                && ops.executing_eagerly_outside_functions();
        }

        private static IRnnCell CreateCell(LSTMArgs lstmArgs)
        {
            return new LSTMCell(new LSTMCellArgs()
            {
                Units = lstmArgs.Units,
                Activation = lstmArgs.Activation,
                RecurrentActivation = lstmArgs.RecurrentActivation,
                UseBias = lstmArgs.UseBias,
                KernelInitializer = lstmArgs.KernelInitializer,
                RecurrentInitializer = lstmArgs.RecurrentInitializer,
                UnitForgetBias = lstmArgs.UnitForgetBias,
                BiasInitializer = lstmArgs.BiasInitializer,
                // TODO(Rinne): kernel_regularizer
                // TODO(Rinne): recurrent_regularizer
                // TODO(Rinne): bias_regularizer
                // TODO(Rinne): kernel_constriant
                // TODO(Rinne): recurrent_constriant
                // TODO(Rinne): bias_constriant
                Dropout = lstmArgs.Dropout,
                RecurrentDropout = lstmArgs.RecurrentDropout,
                Implementation = lstmArgs.Implementation,
                DType = lstmArgs.DType,
                Trainable = lstmArgs.Trainable
            });
        }

        protected override Tensors Call(Tensors inputs, Tensors initial_state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            // skip the condition of ragged input

            (inputs, initial_state, _) = _process_inputs(inputs, initial_state, null);

            Tensor mask = null;
            if(optional_args is RnnOptionalArgs rnnArgs)
            {
                mask = rnnArgs.Mask;
            }

            var single_input = inputs.Single;
            var input_shape = single_input.shape;
            var timesteps = _args.TimeMajor ? input_shape[0] : input_shape[1];

            _maybe_reset_cell_dropout_mask(Cell);

            Func<Tensors, Tensors, (Tensors, Tensors)> step = (inputs, states) =>
            {
                var res = Cell.Apply(inputs, states, training is null ? true : training.Value);
                var (output, state) = res;
                return (output, state);
            };

            var (last_output, outputs, states) = keras.backend.rnn(
                step,
                inputs,
                initial_state,
                constants: null,
                go_backwards: _args.GoBackwards,
                mask: mask,
                unroll: _args.Unroll,
                input_length: ops.convert_to_tensor(timesteps),
                time_major: _args.TimeMajor,
                zero_output_for_mask: _args.ZeroOutputForMask,
                return_all_outputs: _args.ReturnSequences
            );

            Tensor output;
            if (_args.ReturnSequences)
            {
                output = keras.backend.maybe_convert_to_ragged(false, outputs, (int)timesteps, _args.GoBackwards);
            }
            else
            {
                output = last_output;
            }

            if (_args.ReturnState)
            {
                return new Tensor[] { output }.Concat(states).ToArray().ToTensors();
            }
            else
            {
                return output;
            }
        }

        public override IKerasConfig get_config()
        {
            return _args;
        }

    }
}
