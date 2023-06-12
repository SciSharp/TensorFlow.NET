using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using Tensorflow.Common.Extensions;
using Tensorflow.Common.Types;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.ArgsDefinition.Rnn;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Saving;
using Tensorflow.Keras.Utils;

namespace Tensorflow.Keras.Layers.Rnn
{
    public class StackedRNNCells : Layer, IRnnCell
    {
        public IList<IRnnCell> Cells { get; set; }
        public bool reverse_state_order;

        public StackedRNNCells(StackedRNNCellsArgs args) : base(args)
        {
            if (args.Kwargs == null)
            {
                args.Kwargs = new Dictionary<string, object>();
            }
            foreach (var cell in args.Cells)
            {
                //Type type = cell.GetType();
                //var CallMethodInfo = type.GetMethod("Call");
                //if (CallMethodInfo == null)
                //{
                //    throw new ValueError(
                //        "All cells must have a `Call` method. " +
                //    $"Received cell without a `Call` method: {cell}");
                //}
            }
            Cells = args.Cells;
            
            reverse_state_order = (bool)args.Kwargs.Get("reverse_state_order", false);

            if (reverse_state_order)
            {
                throw new WarningException("reverse_state_order=True in StackedRNNCells will soon " +
                                           "be deprecated. Please update the code to work with the " +
                                           "natural order of states if you rely on the RNN states, " +
                                           "eg RNN(return_state=True).");
            }
        }

        public GeneralizedTensorShape StateSize
        {
            get
            {
                GeneralizedTensorShape state_size = new GeneralizedTensorShape(1, Cells.Count);
                if (reverse_state_order && Cells.Count > 0)
                {
                    var idxAndCell = Cells.Reverse().Select((cell, idx) => (idx, cell));
                    foreach (var cell in idxAndCell)
                    {
                        state_size.Shapes[cell.idx] = cell.cell.StateSize.Shapes.First();
                    }
                }
                else
                {
                    //foreach (var cell in Cells)
                    //{
                    //    state_size.Shapes.add(cell.StateSize.Shapes.First());

                    //}
                    var idxAndCell = Cells.Select((cell, idx) => (idx, cell));
                    foreach (var cell in idxAndCell)
                    {
                        state_size.Shapes[cell.idx] = cell.cell.StateSize.Shapes.First();
                    }
                }
                return state_size;
            }
        }

        public object output_size
        {
            get
            {
                var lastCell = Cells.LastOrDefault();
                if (lastCell.OutputSize.ToSingleShape() != -1)
                {
                    return lastCell.OutputSize;
                }
                else if (RNN.is_multiple_state(lastCell.StateSize))
                {
                    return lastCell.StateSize.First();
                    //throw new NotImplementedException("");
                }
                else
                {
                    return lastCell.StateSize;
                }
            }
        }

        public Tensors get_initial_state(Tensors inputs = null, long? batch_size = null, TF_DataType? dtype = null)
        {
            var cells = reverse_state_order ? Cells.Reverse() : Cells;
            Tensors initial_states = new Tensors();
            foreach (var cell in cells)
            {
                var get_initial_state_fn = cell.GetType().GetMethod("get_initial_state");
                if (get_initial_state_fn != null)
                {
                    var result = (Tensors)get_initial_state_fn.Invoke(cell, new object[] { inputs, batch_size, dtype });
                    initial_states.Add(result);
                }
                else
                {
                    initial_states.Add(RnnUtils.generate_zero_filled_state_for_cell(cell, inputs, batch_size.Value, dtype.Value));
                }
            }
            return initial_states;
        }

        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            // Recover per-cell states.
            var state_size = reverse_state_order ? StateSize.Reverse() : StateSize;
            var nested_states = reverse_state_order ? state.Flatten().Reverse() : state.Flatten();


            var new_nest_states = new Tensors();
            // Call the cells in order and store the returned states.
            foreach (var (cell, states) in zip(Cells, nested_states))
            {
                // states = states if tf.nest.is_nested(states) else [states]
                var type = cell.GetType();
                bool IsTFRnnCell = type.GetProperty("IsTFRnnCell") != null;
                state = len(state) == 1 && IsTFRnnCell ? state.FirstOrDefault() : state;

                RnnOptionalArgs? rnn_optional_args = optional_args as RnnOptionalArgs;
                Tensors? constants = rnn_optional_args?.Constants;

                Tensors new_states;
                 (inputs, new_states) = cell.Apply(inputs, states, optional_args: new RnnOptionalArgs() { Constants = constants });

                new_nest_states.Add(new_states);
            }
            new_nest_states = reverse_state_order ? new_nest_states.Reverse().ToArray() : new_nest_states.ToArray();
            return new Nest<Tensor>(new List<Nest<Tensor>> {
                    new Nest<Tensor>(new List<Nest<Tensor>> { new Nest<Tensor>(inputs.Single()) }), new Nest<Tensor>(new_nest_states) })
                    .ToTensors();
        }
        
        

        public void build()
        {
            built = true;
            //  @tf_utils.shape_type_conversion
            //  def build(self, input_shape) :
            //    if isinstance(input_shape, list) :
            //      input_shape = input_shape[0]
            //    for cell in self.cells:
            //      if isinstance(cell, Layer) and not cell.built:
            //        with K.name_scope(cell.name):
            //          cell.build(input_shape)
            //          cell.built = True
            //      if getattr(cell, 'output_size', None) is not None:
            //        output_dim = cell.output_size
            //      elif _is_multiple_state(cell.state_size) :
            //        output_dim = cell.state_size[0]
            //      else:
            //        output_dim = cell.state_size
            //      input_shape = tuple([input_shape[0]] +
            //                          tensor_shape.TensorShape(output_dim).as_list())
            //    self.built = True
        }

        public override IKerasConfig get_config()
        {
            throw new NotImplementedException();
            //def get_config(self):
            //  cells = []
            //  for cell in self.cells:
            //    cells.append(generic_utils.serialize_keras_object(cell))
            //  config = {'cells': cells}
            //  base_config = super(StackedRNNCells, self).get_config()
            //  return dict(list(base_config.items()) + list(config.items()))
        }


        public void from_config()
        {
            throw new NotImplementedException();
            //  @classmethod
            //  def from_config(cls, config, custom_objects = None):
            //    from tensorflow.python.keras.layers import deserialize as deserialize_layer  # pylint: disable=g-import-not-at-top
            //    cells = []
            //    for cell_config in config.pop('cells'):
            //      cells.append(
            //          deserialize_layer(cell_config, custom_objects = custom_objects))
            //    return cls(cells, **config)
        }

        public (Tensor, Tensors) Call(Tensors inputs, Tensors states, bool? training = null)
        {
            throw new NotImplementedException();
        }

        public GeneralizedTensorShape OutputSize => throw new NotImplementedException();
        public bool IsTFRnnCell => true;
        public bool SupportOptionalArgs => throw new NotImplementedException();
    }
}
