using System;
using System.Collections;
using System.Collections.Generic;
using System.Reflection;
using static Tensorflow.Keras.ArgsDefinition.Rnn.RNNArgs;
using Tensorflow.Keras.ArgsDefinition.Rnn;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Saving;
using Tensorflow.Util;
using OneOf;
using OneOf.Types;
using Tensorflow.Common.Extensions;
// from tensorflow.python.distribute import distribution_strategy_context as ds_context;

namespace Tensorflow.Keras.Layers.Rnn
{
    public class RNN : Layer
    {
        private RNNArgs args;
        private object input_spec = null; // or NoneValue??
        private object state_spec = null;
        private Tensors _states = null;
        private object constants_spec = null;
        private int _num_constants = 0;
        protected IVariableV1 kernel;
        protected IVariableV1 bias;
        protected ILayer cell;

        public RNN(RNNArgs args) : base(PreConstruct(args))
        {
            this.args = args;
            SupportsMasking = true;

            // if is StackedRnncell
            if (args.Cell.IsT0)
            {
                cell = new StackedRNNCells(new StackedRNNCellsArgs
                {
                    Cells = args.Cell.AsT0,
                });
            }
            else
            {
                cell = args.Cell.AsT1;
            }

            Type type = cell.GetType();
            MethodInfo callMethodInfo = type.GetMethod("Call");
            if (callMethodInfo == null)
            {
                throw new ValueError(@"Argument `cell` or `cells`should have a `call` method. ");
            }

            PropertyInfo state_size_info = type.GetProperty("state_size");
            if (state_size_info == null)
            {
                throw new ValueError(@"The RNN cell should have a `state_size` attribute");
            }



            // get input_shape
            this.args = PreConstruct(args);
            // The input shape is unknown yet, it could have nested tensor inputs, and
            // the input spec will be the list of specs for nested inputs, the structure
            // of the input_spec will be the same as the input.

            //if(stateful)
            //{
            //    if (ds_context.has_strategy()) // ds_context????
            //    {
            //        throw new Exception("RNNs with stateful=True not yet supported with tf.distribute.Strategy");
            //    }
            //}
        }

        // States is a tuple consist of cell states_size, like (cell1.state_size, cell2.state_size,...)
        // state_size can be a single integer, can also be a list/tuple of integers, can also be TensorShape or a list/tuple of TensorShape
        public Tensors States
        {
            get
            {
                if (_states == null)
                {
                    var state = nest.map_structure(x => null, cell.state_size);
                    return nest.is_nested(state) ? state : new Tensors { state };
                }
                return _states;
            }
            set { _states = value; }
        }

        private OneOf<Shape, List<Shape>> compute_output_shape(Shape input_shape)
        {
            var batch = input_shape[0];
            var time_step = input_shape[1];
            if (args.TimeMajor)
            {
                (batch, time_step) = (time_step, batch);
            }

            // state_size is a array of ints or a positive integer
            var state_size = cell.state_size;

            // TODO(wanglongzhi2001),flat_output_size应该是什么类型的，Shape还是Tensor
            Func<Shape, Shape> _get_output_shape;
            _get_output_shape = (flat_output_size) =>
            {
                var output_dim = flat_output_size.as_int_list();
                Shape output_shape;
                if (args.ReturnSequences)
                {
                    if (args.TimeMajor)
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

            Type type = cell.GetType();
            PropertyInfo output_size_info = type.GetProperty("output_size");
            Shape output_shape;
            if (output_size_info != null)
            {
                output_shape = nest.map_structure(_get_output_shape, cell.output_size);
                // TODO(wanglongzhi2001),output_shape应该简单的就是一个元组还是一个Shape类型
                output_shape = (output_shape.Length == 1 ? (int)output_shape[0] : output_shape);
            }
            else
            {
                output_shape = _get_output_shape(state_size[0]);
            }

            if (args.ReturnState)
            {
                Func<Shape, Shape> _get_state_shape;
                _get_state_shape = (flat_state) =>
                {
                    var state_shape = new int[] { (int)batch }.concat(flat_state.as_int_list());
                    return new Shape(state_shape);
                };
                var state_shape = _get_state_shape(new Shape(state_size.ToArray()));

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
            var output_mask = args.ReturnSequences ? mask : null;
            if (args.ReturnState)
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

                var (batch_index, time_step_index) = args.TimeMajor ? (1, 0) : (0, 1);
                if (!args.Stateful)
                {
                    input_spec_shape[batch_index] = -1;
                }
                input_spec_shape[time_step_index] = -1;
                return new InputSpec(shape: input_spec_shape);
            }

            Shape get_step_input_shape(Shape shape)
            {

                // return shape[1:] if self.time_major else (shape[0],) + shape[2:]
                if (args.TimeMajor)
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


            if (!cell.Built)
            {
                cell.build(input_shape);
            }
        }

        // inputs: Tensors
        // mask: Binary tensor of shape [batch_size, timesteps] indicating whether a given timestep should be masked
        // training: bool
        // initial_state: List of initial state tensors to be passed to the first call of the cell
        // constants: List of constant tensors to be passed to the cell at each timestep
        protected override Tensors Call(Tensors inputs, Tensor mask = null, bool? training = null, Tensors initial_state = null, Tensors constants = null)
        {
            //var (inputs_padded, row_length) = BackendImpl.convert_inputs_if_ragged(inputs);
            // 暂时先不接受ragged tensor
            int? row_length = null;
            bool is_ragged_input = false;
            _validate_args_if_ragged(is_ragged_input, mask);

            (inputs, initial_state, constants) = _process_inputs(inputs, initial_state, constants);

            _maybe_reset_cell_dropout_mask(cell);
            if (cell is StackedRNNCells)
            {
                var stack_cell = cell as StackedRNNCells;
                foreach (var cell in stack_cell.Cells)
                {
                    _maybe_reset_cell_dropout_mask(cell);
                }
            }

            if (mask != null)
            {
                // Time step masks must be the same for each input.
                mask = nest.flatten(mask)[0];
            }

            Shape input_shape;
            if (nest.is_nested(inputs))
            {
                // In the case of nested input, use the first element for shape check
                // input_shape = nest.flatten(inputs)[0].shape;
                // TODO(Wanglongzhi2001)
                input_shape = nest.flatten(inputs)[0].shape;
            }
            else
            {
                input_shape = inputs.shape;
            }

            var timesteps = args.TimeMajor ? input_shape[0] : input_shape[1];

            if (args.Unroll && timesteps != null)
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
            var cell_call_fn = cell.Call;
            Func<Tensors, Tensors, (Tensors, Tensors)> step;
            if (constants != null)
            {
                ParameterInfo[] parameters = cell_call_fn.GetMethodInfo().GetParameters();
                bool hasParam = parameters.Any(p => p.Name == "constants");
                if (!hasParam)
                {
                    throw new ValueError(
                          $"RNN cell {cell} does not support constants." +
                          $"Received: constants={constants}");
                }

                step = (inputs, states) =>
                {
                    // constants = states[-self._num_constants :]
                    constants = states.numpy()[new Slice(states.Length - _num_constants, states.Length)];
                    // states = states[: -self._num_constants]
                    states = states.numpy()[new Slice(0, states.Length - _num_constants)];
                    // states = (states[0] if len(states) == 1 and is_tf_rnn_cell else states)
                    states = states.Length == 1 ? states[0] : states;
                    var (output, new_states) = cell_call_fn(inputs, null, null, states, constants);
                    // TODO(Wanglongzhi2001),should cell_call_fn's return value be Tensors, Tensors?
                    if (!nest.is_nested(new_states))
                    {
                        return (output, new Tensors { new_states });
                    }
                    return (output, new_states);
                };
            }
            else
            {
                step = (inputs, states) =>
                {
                    // states = (states[0] if len(states) == 1 and is_tf_rnn_cell else states)
                    states = states.Length == 1 ? states[0] : states;
                    var (output, new_states) = cell_call_fn(inputs, null, null, states, constants);
                    if (!nest.is_nested(new_states))
                    {
                        return (output, new Tensors { new_states });
                    }
                    return (output, new_states);
                };
            }

            var (last_output, outputs, states) = BackendImpl.rnn(step,
                inputs,
                initial_state,
                constants: constants,
                go_backwards: args.GoBackwards,
                mask: mask,
                unroll: args.Unroll,
                input_length: row_length != null ? new Tensor(row_length) : new Tensor(timesteps),
                time_major: args.TimeMajor,
                zero_output_for_mask: args.ZeroOutputForMask,
                return_all_outputs: args.ReturnSequences);

            if (args.Stateful)
            {
                throw new NotImplementedException("this argument havn't been developed!");
            }

            Tensors output = new Tensors();
            if (args.ReturnSequences)
            {
                throw new NotImplementedException("this argument havn't been developed!");

            }
            else
            {
                output = last_output;
            }

            if (args.ReturnState)
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

        private (Tensors inputs, Tensors initial_state, Tensors constants) _process_inputs(Tensor inputs, Tensors initial_state, Tensors constants)
        {
            if (nest.is_sequence(input))
            {
                if (_num_constants != 0)
                {
                    initial_state = inputs[new Slice(1, len(inputs))];
                }
                else
                {
                    initial_state = inputs[new Slice(1, len(inputs) - _num_constants)];
                    constants = inputs[new Slice(len(inputs) - _num_constants, len(inputs))];
                }
                if (len(initial_state) == 0)
                    initial_state = null;
                inputs = inputs[0];
            }

            if (args.Stateful)
            {
                if (initial_state != null)
                {
                    var tmp = new Tensor[] { };
                    foreach (var s in nest.flatten(States))
                    {
                        tmp.add(tf.math.count_nonzero((Tensor)s));
                    }
                    var non_zero_count = tf.add_n(tmp);
                    //initial_state = tf.cond(non_zero_count > 0, () => States, () => initial_state);
                    if((int)non_zero_count.numpy() > 0)
                    {
                        initial_state = States;
                    }
                }
                else
                {
                    initial_state = States;
                }

            }
            else if(initial_state != null)
            {
                initial_state = get_initial_state(inputs);
            }

            if (initial_state.Length != States.Length)
            {
                throw new ValueError(
                                       $"Layer {this} expects {States.Length} state(s), " +
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

            if (args.Unroll)
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
            //if (cell is DropoutRNNCellMixin)
            //{
            //    cell.reset_dropout_mask();
            //    cell.reset_recurrent_dropout_mask();
            //}
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


        protected Tensors get_initial_state(Tensor inputs)
        {
            Type type = cell.GetType();
            MethodInfo MethodInfo = type.GetMethod("get_initial_state");

            if (nest.is_nested(inputs))
            {
                // The input are nested sequences. Use the first element in the seq
                // to get batch size and dtype.
                inputs = nest.flatten(inputs)[0];
            }

            var input_shape = tf.shape(inputs);
            var batch_size = args.TimeMajor ? input_shape[1] : input_shape[0];
            var dtype = inputs.dtype;
            Tensor init_state;
            if (MethodInfo != null)
            {
                init_state = (Tensor)MethodInfo.Invoke(cell, new object[] { null, batch_size, dtype });
            }
            else
            {
                init_state = RNNUtils.generate_zero_filled_state(batch_size, cell.state_size, dtype);
            }

            //if (!nest.is_nested(init_state))
            //{
            //    init_state = new List<Tensor> { init_state};
            //}
            return new List<Tensor> { init_state };

            //return _generate_zero_filled_state_for_cell(null, null);
        }

        Tensor _generate_zero_filled_state_for_cell(LSTMCell cell, Tensor batch_size)
        {
            throw new NotImplementedException("");
        }

        // Check whether the state_size contains multiple states.
        public static bool is_multiple_state(object state_size)
        {
            var myIndexerProperty = state_size.GetType().GetProperty("Item");
            return myIndexerProperty != null
                && myIndexerProperty.GetIndexParameters().Length == 1
                && !(state_size.GetType() == typeof(Shape));
        }
    }
}
