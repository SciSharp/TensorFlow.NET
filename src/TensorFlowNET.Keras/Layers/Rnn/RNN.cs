using OneOf;
using System;
using System.Collections.Generic;
using System.Reflection;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Saving;
using Tensorflow.Util;
using Tensorflow.Common.Extensions;
using System.Linq.Expressions;
using Tensorflow.Keras.Utils;
using Tensorflow.Common.Types;
using System.Runtime.CompilerServices;
// from tensorflow.python.distribute import distribution_strategy_context as ds_context;

namespace Tensorflow.Keras.Layers
{
    /// <summary>
    /// Base class for recurrent layers.
    /// See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
    /// for details about the usage of RNN API.
    /// </summary>
    public class RNN : RnnBase
    {
        private RNNArgs _args;
        private object _input_spec = null; // or NoneValue??
        private object _state_spec = null;
        private object _constants_spec = null;
        private Tensors _states = null;
        private int _num_constants;
        protected IVariableV1 _kernel;
        protected IVariableV1 _bias;
        private IRnnCell _cell;

        public RNNArgs Args { get => _args; }
        public IRnnCell Cell
        {
            get
            {
                return _cell;
            }
            init
            {
                _cell = value;
                _self_tracked_trackables.Add(_cell);
            }
        }

        public RNN(IRnnCell cell, RNNArgs args) : base(PreConstruct(args))
        {
            _args = args;
            SupportsMasking = true;

            Cell = cell;

            // get input_shape
            _args = PreConstruct(args);

            _num_constants = 0;
        }

        public RNN(IEnumerable<IRnnCell> cells, RNNArgs args) : base(PreConstruct(args))
        {
            _args = args;
            SupportsMasking = true;

            Cell = new StackedRNNCells(cells, new StackedRNNCellsArgs());

            // get input_shape
            _args = PreConstruct(args);

            _num_constants = 0;
        }

        // States is a tuple consist of cell states_size, like (cell1.state_size, cell2.state_size,...)
        // state_size can be a single integer, can also be a list/tuple of integers, can also be TensorShape or a list/tuple of TensorShape
        public Tensors States
        {
            get
            {
                if (_states == null)
                {
                    // CHECK(Rinne): check if this is correct.
                    var nested = Cell.StateSize.MapStructure<Tensor?>(x => null);
                    _states = nested.AsNest().ToTensors();
                }
                return _states;
            }
            set { _states = value; }
        }

        private INestStructure<Shape> compute_output_shape(Shape input_shape)
        {
            var batch = input_shape[0];
            var time_step = input_shape[1];
            if (_args.TimeMajor)
            {
                (batch, time_step) = (time_step, batch);
            }

            // state_size is a array of ints or a positive integer
            var state_size = Cell.StateSize;
            if(state_size?.TotalNestedCount == 1)
            {
                state_size = new NestList<long>(state_size.Flatten().First());
            }

            Func<long, Shape>  _get_output_shape = (flat_output_size) =>
            {
                var output_dim = new Shape(flat_output_size).as_int_list();
                Shape output_shape;
                if (_args.ReturnSequences)
                {
                    if (_args.TimeMajor)
                    {
                        output_shape = new Shape(new int[] { (int)time_step, (int)batch }.concat(output_dim));
                    }
                    else
                    {
                        output_shape = new Shape(new int[] { (int)batch, (int)time_step }.concat(output_dim));

                    }
                }
                else
                {
                    output_shape = new Shape(new int[] { (int)batch }.concat(output_dim));
                }
                return output_shape;
            };

            Type type = Cell.GetType();
            PropertyInfo output_size_info = type.GetProperty("output_size");
            INestStructure<Shape> output_shape;
            if (output_size_info != null)
            {
                output_shape = Nest.MapStructure(_get_output_shape, Cell.OutputSize);
            }
            else
            {
                output_shape = new NestNode<Shape>(_get_output_shape(state_size.Flatten().First()));
            }

            if (_args.ReturnState)
            {
                Func<long, Shape> _get_state_shape = (flat_state) =>
                {
                    var state_shape = new int[] { (int)batch }.concat(new Shape(flat_state).as_int_list());
                    return new Shape(state_shape);
                };


                var state_shape = Nest.MapStructure(_get_state_shape, state_size);

                return new Nest<Shape>(new[] { output_shape, state_shape } );
            }
            else
            {
                return output_shape;
            }

        }

        private Tensors compute_mask(Tensors inputs, Tensors mask)
        {
            // Time step masks must be the same for each input.
            // This is because the mask for an RNN is of size [batch, time_steps, 1],
            // and specifies which time steps should be skipped, and a time step
            // must be skipped for all inputs.

            mask = nest.flatten(mask)[0];
            var output_mask = _args.ReturnSequences ? mask : null;
            if (_args.ReturnState)
            {
                var state_mask = new List<Tensor>();
                for (int i = 0; i < len(States); i++)
                {
                    state_mask.Add(null);
                }
                return new List<Tensor> { output_mask }.concat(state_mask);
            }
            else
            {
                return output_mask;
            }
        }

        public override void build(KerasShapesWrapper input_shape)
        {
            _buildInputShape = input_shape;
            input_shape = new KerasShapesWrapper(input_shape.Shapes[0]);

            InputSpec get_input_spec(Shape shape)
            {
                var input_spec_shape = shape.as_int_list();

                var (batch_index, time_step_index) = _args.TimeMajor ? (1, 0) : (0, 1);
                if (!_args.Stateful)
                {
                    input_spec_shape[batch_index] = -1;
                }
                input_spec_shape[time_step_index] = -1;
                return new InputSpec(shape: input_spec_shape);
            }

            Shape get_step_input_shape(Shape shape)
            {

                // return shape[1:] if self.time_major else (shape[0],) + shape[2:]
                if (_args.TimeMajor)
                {
                    return shape.as_int_list().ToList().GetRange(1, shape.Length - 1).ToArray();
                }
                else
                {
                    return new int[] { shape.as_int_list()[0] }.concat(shape.as_int_list().ToList().GetRange(2, shape.Length - 2).ToArray());
                }


            }

            object get_state_spec(Shape shape)
            {
                var state_spec_shape = shape.as_int_list();
                // append bacth dim
                state_spec_shape = new int[] { -1 }.concat(state_spec_shape);
                return new InputSpec(shape: state_spec_shape);
            }

            // Check whether the input shape contains any nested shapes. It could be
            // (tensor_shape(1, 2), tensor_shape(3, 4)) or (1, 2, 3) which is from
            // numpy inputs.


            if (Cell is Layer layer && !layer.Built)
            {
                layer.build(input_shape);
                layer.Built = true;
            }

            this.built = true;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="initial_state">List of initial state tensors to be passed to the first call of the cell</param>
        /// <param name="training"></param>
        /// <param name="optional_args"></param>
        /// <returns></returns>
        /// <exception cref="ValueError"></exception>
        /// <exception cref="NotImplementedException"></exception>
        protected override Tensors Call(Tensors inputs, Tensors initial_state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            RnnOptionalArgs? rnn_optional_args = optional_args as RnnOptionalArgs;
            if(optional_args is not null && rnn_optional_args is null)
            {
                throw new ArgumentException("The optional args shhould be of type `RnnOptionalArgs`");
            }
            Tensors? constants = rnn_optional_args?.Constants;
            Tensors? mask = rnn_optional_args?.Mask;
            //var (inputs_padded, row_length) = BackendImpl.convert_inputs_if_ragged(inputs);
            // 暂时先不接受ragged tensor
            int row_length = 0; // TODO(Rinne): support this param.
            bool is_ragged_input = false;
            _validate_args_if_ragged(is_ragged_input, mask);

            (inputs, initial_state, constants) = _process_inputs(inputs, initial_state, constants);

            _maybe_reset_cell_dropout_mask(Cell);
            if (Cell is StackedRNNCells)
            {
                var stack_cell = Cell as StackedRNNCells;
                foreach (IRnnCell cell in stack_cell.Cells)
                {
                    _maybe_reset_cell_dropout_mask(cell);
                }
            }

            if (mask != null)
            {
                // Time step masks must be the same for each input.
                mask = mask.Flatten().First();
            }

            Shape input_shape;
            if (!inputs.IsNested())
            {
                // In the case of nested input, use the first element for shape check
                // input_shape = nest.flatten(inputs)[0].shape;
                // TODO(Wanglongzhi2001)
                input_shape = inputs.Flatten().First().shape;
            }
            else
            {
                input_shape = inputs.shape;
            }

            var timesteps = _args.TimeMajor ? input_shape[0] : input_shape[1];

            if (_args.Unroll && timesteps == null)
            {
                throw new ValueError(
                "Cannot unroll a RNN if the " +
                "time dimension is undefined. \n" +
                "- If using a Sequential model, " +
                "specify the time dimension by passing " +
                "an `input_shape` or `batch_input_shape` " +
                "argument to your first layer. If your " +
                "first layer is an Embedding, you can " +
                "also use the `input_length` argument.\n" +
                "- If using the functional API, specify " +
                "the time dimension by passing a `shape` " +
                "or `batch_shape` argument to your Input layer."
                );
            }

            // cell_call_fn = (self.cell.__call__ if callable(self.cell) else self.cell.call)
            Func<Tensors, Tensors, (Tensors, Tensors)> step;
            bool is_tf_rnn_cell = false;
            if (constants is not null)
            {
                if (!Cell.SupportOptionalArgs)
                {
                    throw new ValueError(
                          $"RNN cell {Cell} does not support constants." +
                          $"Received: constants={constants}");
                }

                step = (inputs, states) =>
                {
                    constants = new Tensors(states.TakeLast(_num_constants).ToArray());
                    states = new Tensors(states.SkipLast(_num_constants).ToArray());
                    states = len(states) == 1 && is_tf_rnn_cell ? new Tensors(states[0]) : states;
                    var (output, new_states) = Cell.Apply(inputs, states, optional_args: new RnnOptionalArgs() { Constants = constants });
                    return (output, new_states);
                };
            }
            else
            {
                step = (inputs, states) =>
                {
                    states = len(states) == 1 && is_tf_rnn_cell ? new Tensors(states.First()) : states;
                    var (output, new_states) = Cell.Apply(inputs, states);
                    return (output, new_states);
                };
            }
           
            var (last_output, outputs, states) = keras.backend.rnn(
                step,
                inputs,
                initial_state,
                constants: constants,
                go_backwards: _args.GoBackwards,
                mask: mask,
                unroll: _args.Unroll,
                input_length: row_length != null ? new Tensor(row_length) : new Tensor(timesteps),
                time_major: _args.TimeMajor,
                zero_output_for_mask: _args.ZeroOutputForMask,
                return_all_outputs: _args.ReturnSequences);

            if (_args.Stateful)
            {
                throw new NotImplementedException("this argument havn't been developed.");
            }

            Tensors output = new Tensors();
            if (_args.ReturnSequences)
            {
                // TODO(Rinne): add go_backwards parameter and revise the `row_length` param
                output = keras.backend.maybe_convert_to_ragged(is_ragged_input, outputs, row_length, false);
            }
            else
            {
                output = last_output;
            }

            if (_args.ReturnState)
            {
                foreach (var state in states)
                {
                    output.Add(state);
                }
                return output;
            }
            else
            {
                //var tapeSet = tf.GetTapeSet();
                //foreach(var tape in tapeSet)
                //{
                //    tape.Watch(output);
                //}
                return output;
            }
        }

        public override Tensors Apply(Tensors inputs, Tensors initial_states = null, bool? training = false, IOptionalArgs? optional_args = null)
        {
            RnnOptionalArgs? rnn_optional_args = optional_args as RnnOptionalArgs;
            if (optional_args is not null && rnn_optional_args is null)
            {
                throw new ArgumentException("The type of optional args should be `RnnOptionalArgs`.");
            }
            Tensors? constants = rnn_optional_args?.Constants;
            (inputs, initial_states, constants) = RnnUtils.standardize_args(inputs, initial_states, constants, _num_constants);

            if(initial_states is null && constants is null)
            {
                return base.Apply(inputs);
            }

            // TODO(Rinne): implement it.
            throw new NotImplementedException();
        }

        protected (Tensors inputs, Tensors initial_state, Tensors constants) _process_inputs(Tensors inputs, Tensors initial_state, Tensors constants)
        {
            if (inputs.Length > 1)
            {
                if (_num_constants != 0)
                {
                    initial_state = new Tensors(inputs.Skip(1).ToArray());
                }
                else
                {
                    initial_state = new Tensors(inputs.Skip(1).SkipLast(_num_constants).ToArray());
                    constants = new Tensors(inputs.TakeLast(_num_constants).ToArray());
                }
                if (len(initial_state) == 0)
                    initial_state = null;
                inputs = inputs[0];
            }
            

            if (_args.Stateful)
            {
                if (initial_state != null)
                {
                    var tmp = new Tensor[] { };
                    foreach (var s in nest.flatten(States))
                    {
                        tmp.add(tf.math.count_nonzero(s.Single()));
                    }
                    var non_zero_count = tf.add_n(tmp);
                    initial_state = tf.cond(non_zero_count > 0, States, initial_state);
                    if ((int)non_zero_count.numpy() > 0)
                    {
                        initial_state = States;
                    }
                }
                else
                {
                    initial_state = States;
                }
                //initial_state = Nest.MapStructure(v => tf.cast(v, this.), initial_state);
            }
            else if (initial_state is null)
            {
                initial_state = get_initial_state(inputs);
            }

            if (initial_state.Length != States.Length)
            {
                throw new ValueError($"Layer {this} expects {States.Length} state(s), " +
                                     $"but it received {initial_state.Length} " +
                                     $"initial state(s). Input received: {inputs}");
            }

            return (inputs, initial_state, constants);
        }

        protected void _validate_args_if_ragged(bool is_ragged_input, Tensors mask)
        {
            if (!is_ragged_input)
            {
                return;
            }

            if (_args.Unroll)
            {
                throw new ValueError("The input received contains RaggedTensors and does " +
                "not support unrolling. Disable unrolling by passing " +
                "`unroll=False` in the RNN Layer constructor.");
            }
            if (mask != null)
            {
                throw new ValueError($"The mask that was passed in was {mask}, which " +
                "cannot be applied to RaggedTensor inputs. Please " +
                "make sure that there is no mask injected by upstream " +
                "layers.");
            }

        }

        protected void _maybe_reset_cell_dropout_mask(ILayer cell)
        {
            if (cell is DropoutRNNCellMixin CellDRCMixin)
            {
                CellDRCMixin.reset_dropout_mask();
                CellDRCMixin.reset_recurrent_dropout_mask();
            }
        }

        private static RNNArgs PreConstruct(RNNArgs args)
        {
            // If true, the output for masked timestep will be zeros, whereas in the
            // false case, output from previous timestep is returned for masked timestep.
            var zeroOutputForMask = args.ZeroOutputForMask;

            Shape input_shape;
            var propIS = args.InputShape;
            var propID = args.InputDim;
            var propIL = args.InputLength;

            if (propIS == null && (propID != null || propIL != null))
            {
                input_shape = new Shape(
                    propIL ?? -1,
                    propID ?? -1);
                args.InputShape = input_shape;
            }

            return args;
        }

        public Tensors __call__(Tensors inputs, Tensor state = null, Tensor training = null)
        {
            throw new NotImplementedException();
        }

        protected Tensors get_initial_state(Tensors inputs)
        {
            var input = inputs[0];
            var input_shape = array_ops.shape(inputs);
            var batch_size = _args.TimeMajor ? input_shape[1] : input_shape[0];
            var dtype = input.dtype;
            Tensors init_state = Cell.GetInitialState(null, batch_size, dtype);
            return init_state;
        }

        public override IKerasConfig get_config()
        {
            return _args;
        }
    }
}
