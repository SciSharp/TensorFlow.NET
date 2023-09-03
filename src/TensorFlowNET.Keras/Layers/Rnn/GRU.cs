using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Common.Extensions;
using Tensorflow.Common.Types;
using Tensorflow.Keras.Saving;


namespace Tensorflow.Keras.Layers
{
    public class GRU : RNN
    {
        GRUArgs _args;
        private static GRUCell _cell;

        bool _return_runtime;
        public GRUCell Cell { get => _cell; }
        public int units { get => _args.Units; }
        public Activation activation { get => _args.Activation; }
        public Activation recurrent_activation { get => _args.RecurrentActivation; }
        public bool use_bias { get => _args.UseBias; }
        public float dropout { get => _args.Dropout; }
        public float recurrent_dropout { get => _args.RecurrentDropout; }
        public IInitializer kernel_initializer { get => _args.KernelInitializer; }
        public IInitializer recurrent_initializer { get => _args.RecurrentInitializer; }
        public IInitializer bias_initializer { get => _args.BiasInitializer; }
        public int implementation { get => _args.Implementation; }
        public bool reset_after { get => _args.ResetAfter; }

        public GRU(GRUArgs args) : base(CreateCell(args), PreConstruct(args))
        {
            _args = args;

            if (_args.Implementation == 0)
            {
                // Use the red output to act as a warning message that can also be used under the release version
                Console.ForegroundColor = ConsoleColor.Red; 
                Console.WriteLine("Warning: `implementation=0` has been deprecated, "+
                    "and now defaults to `implementation=2`."+
                    "Please update your layer call.");
                Console.ResetColor();
            }

            GRUCell cell = new GRUCell(new GRUCellArgs
            {
                Units = _args.Units,
                Activation = _args.Activation,
                RecurrentActivation = _args.RecurrentActivation,
                UseBias = _args.UseBias,
                Dropout = _args.Dropout,
                RecurrentDropout = _args.RecurrentDropout,
                KernelInitializer = _args.KernelInitializer,
                RecurrentInitializer = _args.RecurrentInitializer,
                BiasInitializer = _args.BiasInitializer,
                ResetAfter = _args.ResetAfter,
                Implementation = _args.Implementation
            });
            _cell = cell;
        }

        protected override Tensors Call(Tensors inputs, Tensors initial_state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            GRUOptionalArgs? gru_optional_args = optional_args as GRUOptionalArgs;
            if (optional_args is not null && gru_optional_args is null)
            {
                throw new ArgumentException("The type of optional args should be `GRUOptionalArgs`.");
            }
            Tensors? mask = gru_optional_args?.Mask;

            // Not support ragger input temporarily;
            int row_length = 0;
            bool is_ragged_input = false;

            _validate_args_if_ragged(is_ragged_input, mask);

            // GRU does not support constants.Ignore it during process.
             (inputs, initial_state, _) = this._process_inputs(inputs, initial_state, null);

            if (mask.Length > 1)
            {
                mask = mask[0];
            }

            var input_shape = inputs.shape;
            var timesteps = _args.TimeMajor ? input_shape[0] : input_shape[1];


            // TODO(Wanglongzhi2001), finish _could_use_gpu_kernel part
            Func<Tensors, Tensors, (Tensors, Tensors)> step = (cell_inputs, cell_states) =>
            {
                var res = Cell.Apply(cell_inputs, cell_states, training is null ? true : training.Value);
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
                zero_output_for_mask: base.Args.ZeroOutputForMask,
                return_all_outputs: _args.ReturnSequences
            );

            Tensors output;
            if (_args.ReturnSequences)
            {
                output = outputs;   
            }
            else
            {
                output = last_output;
            }

            if (_args.ReturnState)
            {
                output = new Tensors { output, states };
            }
            return output;
        }

        private static IRnnCell CreateCell(GRUArgs gruArgs)
        {
            return new GRUCell(new GRUCellArgs
            {
                Units = gruArgs.Units,
                Activation = gruArgs.Activation,
                RecurrentActivation = gruArgs.RecurrentActivation,
                UseBias = gruArgs.UseBias,
                Dropout = gruArgs.Dropout,
                RecurrentDropout = gruArgs.RecurrentDropout,
                KernelInitializer = gruArgs.KernelInitializer,
                RecurrentInitializer = gruArgs.RecurrentInitializer,
                BiasInitializer = gruArgs.BiasInitializer,
                ResetAfter = gruArgs.ResetAfter,
                Implementation = gruArgs.Implementation
            });
        }

        private static RNNArgs PreConstruct(GRUArgs args)
        {
            return new RNNArgs
            {
                ReturnSequences = args.ReturnSequences,
                ReturnState = args.ReturnState,
                GoBackwards = args.GoBackwards,
                Stateful = args.Stateful,
                Unroll = args.Unroll,
                TimeMajor = args.TimeMajor,
                Units = args.Units,
                Activation = args.Activation,
                RecurrentActivation = args.RecurrentActivation,
                UseBias = args.UseBias,
                Dropout = args.Dropout,
                RecurrentDropout = args.RecurrentDropout,
                KernelInitializer = args.KernelInitializer,
                RecurrentInitializer = args.RecurrentInitializer,
                BiasInitializer = args.BiasInitializer
            };
        }
    }
}
