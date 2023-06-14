﻿using OneOf;
using System;
using System.Collections.Generic;
using System.Reflection;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.ArgsDefinition.Rnn;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Saving;
using Tensorflow.Util;
using Tensorflow.Common.Extensions;
using System.Linq.Expressions;
using Tensorflow.Keras.Utils;
using Tensorflow.Common.Types;
// from tensorflow.python.distribute import distribution_strategy_context as ds_context;

namespace Tensorflow.Keras.Layers.Rnn
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
        private Tensors _states = null;
        private object _constants_spec = null;
        private int _num_constants;
        protected IVariableV1 _kernel;
        protected IVariableV1 _bias;
        protected IRnnCell _cell;

        public RNN(RNNArgs args) : base(PreConstruct(args))
        {
            _args = args;
            SupportsMasking = true;

            // if is StackedRnncell
            if (args.Cells != null)
            {
                _cell = new StackedRNNCells(new StackedRNNCellsArgs
                {
                    Cells = args.Cells
                });
            }
            else
            {
                _cell = args.Cell;
            }

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
                    var nested = _cell.StateSize.MapStructure<Tensor?>(x => null);
                    _states = nested.AsNest().ToTensors();
                }
                return _states;
            }
            set { _states = value; }
        }

        private OneOf<Shape, List<Shape>> compute_output_shape(Shape input_shape)
        {
            var batch = input_shape[0];
            var time_step = input_shape[1];
            if (_args.TimeMajor)
            {
                (batch, time_step) = (time_step, batch);
            }

            // state_size is a array of ints or a positive integer
            var state_size = _cell.StateSize.ToSingleShape();

            // TODO(wanglongzhi2001),flat_output_size应该是什么类型的，Shape还是Tensor
            Func<Shape, Shape> _get_output_shape;
            _get_output_shape = (flat_output_size) =>
            {
                var output_dim = flat_output_size.as_int_list();
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

            Type type = _cell.GetType();
            PropertyInfo output_size_info = type.GetProperty("output_size");
            Shape output_shape;
            if (output_size_info != null)
            {
                output_shape = nest.map_structure(_get_output_shape, _cell.OutputSize.ToSingleShape());
                // TODO(wanglongzhi2001),output_shape应该简单的就是一个元组还是一个Shape类型
                output_shape = (output_shape.Length == 1 ? (int)output_shape[0] : output_shape);
            }
            else
            {
                output_shape = _get_output_shape(state_size);
            }

            if (_args.ReturnState)
            {
                Func<Shape, Shape> _get_state_shape;
                _get_state_shape = (flat_state) =>
                {
                    var state_shape = new int[] { (int)batch }.concat(flat_state.as_int_list());
                    return new Shape(state_shape);
                };


                var state_shape = _get_state_shape(state_size);

                return new List<Shape> { output_shape, state_shape };
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
            object get_input_spec(Shape shape)
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


            if (!_cell.Built)
            {
                _cell.build(input_shape);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="mask">Binary tensor of shape [batch_size, timesteps] indicating whether a given timestep should be masked</param>
        /// <param name="training"></param>
        /// <param name="initial_state">List of initial state tensors to be passed to the first call of the cell</param>
        /// <param name="constants">List of constant tensors to be passed to the cell at each timestep</param>
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

            _maybe_reset_cell_dropout_mask(_cell);
            if (_cell is StackedRNNCells)
            {
                var stack_cell = _cell as StackedRNNCells;
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
            bool is_tf_rnn_cell = _cell.IsTFRnnCell;
            if (constants is not null)
            {
                if (!_cell.SupportOptionalArgs)
                {
                    throw new ValueError(
                          $"RNN cell {_cell} does not support constants." +
                          $"Received: constants={constants}");
                }

                step = (inputs, states) =>
                {
                    constants = new Tensors(states.TakeLast(_num_constants));
                    states = new Tensors(states.SkipLast(_num_constants));
                    states = len(states) == 1 && is_tf_rnn_cell ? new Tensors(states[0]) : states;
                    var (output, new_states) = _cell.Apply(inputs, states, optional_args: new RnnOptionalArgs() { Constants = constants });
                    return (output, new_states.Single);
                };
            }
            else
            {
                step = (inputs, states) =>
                {
                    states = len(states) == 1 && is_tf_rnn_cell ? new Tensors(states.First()) : states;
                    var (output, new_states) = _cell.Apply(inputs, states);
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
                return output;
            }
        }

        public override Tensors Apply(Tensors inputs, Tensors initial_states = null, bool training = false, IOptionalArgs? optional_args = null)
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

        private (Tensors inputs, Tensors initial_state, Tensors constants) _process_inputs(Tensors inputs, Tensors initial_state, Tensors constants)
        {
            if (inputs.Length > 1)
            {
                if (_num_constants != 0)
                {
                    initial_state = new Tensors(inputs.Skip(1));
                }
                else
                {
                    initial_state = new Tensors(inputs.Skip(1).SkipLast(_num_constants));
                    constants = new Tensors(inputs.TakeLast(_num_constants));
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
                    //initial_state = tf.cond(non_zero_count > 0, () => States, () => initial_state);
                    if ((int)non_zero_count.numpy() > 0)
                    {
                        initial_state = States;
                    }
                }
                else
                {
                    initial_state = States;
                }
                // TODO(Wanglongzhi2001),
//                initial_state = tf.nest.map_structure(
//# When the layer has a inferred dtype, use the dtype from the
//# cell.
//                lambda v: tf.cast(
//                    v, self.compute_dtype or self.cell.compute_dtype
//                ),
//                initial_state,
//            )

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

        private void _validate_args_if_ragged(bool is_ragged_input, Tensors mask)
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

        void _maybe_reset_cell_dropout_mask(ILayer cell)
        {
            if (cell is DropoutRNNCellMixin CellDRCMixin)
            {
                CellDRCMixin.reset_dropout_mask();
                CellDRCMixin.reset_recurrent_dropout_mask();
            }
        }

        private static RNNArgs PreConstruct(RNNArgs args)
        {
            if (args.Kwargs == null)
            {
                args.Kwargs = new Dictionary<string, object>();
            }

            // If true, the output for masked timestep will be zeros, whereas in the
            // false case, output from previous timestep is returned for masked timestep.
            var zeroOutputForMask = (bool)args.Kwargs.Get("zero_output_for_mask", false);

            Shape input_shape;
            var propIS = (Shape)args.Kwargs.Get("input_shape", null);
            var propID = (int?)args.Kwargs.Get("input_dim", null);
            var propIL = (int?)args.Kwargs.Get("input_length", null);

            if (propIS == null && (propID != null || propIL != null))
            {
                input_shape = new Shape(
                    propIL ?? -1,
                    propID ?? -1);
                args.Kwargs["input_shape"] = input_shape;
            }

            return args;
        }

        public Tensors __call__(Tensors inputs, Tensor state = null, Tensor training = null)
        {
            throw new NotImplementedException();
        }

        // 好像不能cell不能传接口类型
        //public RNN New(IRnnArgCell cell,
        //    bool return_sequences = false,
        //    bool return_state = false,
        //    bool go_backwards = false,
        //    bool stateful = false,
        //    bool unroll = false,
        //    bool time_major = false)
        //        => new RNN(new RNNArgs
        //        {
        //            Cell = cell,
        //            ReturnSequences = return_sequences,
        //            ReturnState = return_state,
        //            GoBackwards = go_backwards,
        //            Stateful = stateful,
        //            Unroll = unroll,
        //            TimeMajor = time_major
        //        });

        //public RNN New(List<IRnnArgCell> cell,
        //    bool return_sequences = false,
        //    bool return_state = false,
        //    bool go_backwards = false,
        //    bool stateful = false,
        //    bool unroll = false,
        //    bool time_major = false)
        //        => new RNN(new RNNArgs
        //        {
        //            Cell = cell,
        //            ReturnSequences = return_sequences,
        //            ReturnState = return_state,
        //            GoBackwards = go_backwards,
        //            Stateful = stateful,
        //            Unroll = unroll,
        //            TimeMajor = time_major
        //        });


        protected Tensors get_initial_state(Tensors inputs)
        {
            var get_initial_state_fn = _cell.GetType().GetMethod("get_initial_state");

            var input = inputs[0];
            var input_shape = inputs.shape;
            var batch_size = _args.TimeMajor ? input_shape[1] : input_shape[0];
            var dtype = input.dtype;

            Tensors init_state = new Tensors();

            if(get_initial_state_fn != null)
            {
                init_state = (Tensors)get_initial_state_fn.Invoke(_cell, new object[] { inputs, batch_size, dtype });
                
            }
            //if (_cell is RnnCellBase rnn_base_cell)
            //{
            //    init_state = rnn_base_cell.GetInitialState(null, batch_size, dtype);
            //}
            else
            {
                init_state = RnnUtils.generate_zero_filled_state(batch_size, _cell.StateSize, dtype);
            }

            return init_state;
        }

        // Check whether the state_size contains multiple states.
        public static bool is_multiple_state(GeneralizedTensorShape state_size)
        {
            return state_size.Shapes.Length > 1;
        }
    }
}
