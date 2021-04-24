using System;
using System.Collections.Generic;
using System.ComponentModel;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;

namespace Tensorflow.Keras.Layers
{
    public class StackedRNNCells : Layer, RNNArgs.IRnnArgCell
    {
        public IList<RnnCell> Cells { get; set; }
        public bool reverse_state_order;

        public StackedRNNCells(StackedRNNCellsArgs args) : base(args)
        {
            if (args.Kwargs == null)
            {
                args.Kwargs = new Dictionary<string, object>();
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

        public object state_size
        {
            get => throw new NotImplementedException();
            //@property
            //def state_size(self) :
            //    return tuple(c.state_size for c in
            //                 (self.cells[::- 1] if self.reverse_state_order else self.cells))
        }

        public object output_size
        {
            get
            {
                var lastCell = Cells[Cells.Count - 1];

                if (lastCell.output_size != -1)
                {
                    return lastCell.output_size;
                }
                else if (RNN._is_multiple_state(lastCell.state_size))
                {
                    return ((dynamic)Cells[-1].state_size)[0];
                }
                else
                {
                    return Cells[-1].state_size;
                }
            }
        }

        public object get_initial_state()
        {
            throw new NotImplementedException();
            //  def get_initial_state(self, inputs= None, batch_size= None, dtype= None) :
            //    initial_states = []
            //    for cell in self.cells[::- 1] if self.reverse_state_order else self.cells:
            //      get_initial_state_fn = getattr(cell, 'get_initial_state', None)
            //      if get_initial_state_fn:
            //        initial_states.append(get_initial_state_fn(
            //            inputs=inputs, batch_size=batch_size, dtype=dtype))
            //      else:
            //        initial_states.append(_generate_zero_filled_state_for_cell(
            //            cell, inputs, batch_size, dtype))

            //    return tuple(initial_states)
        }

        public object call()
        {
            throw new NotImplementedException();
            //  def call(self, inputs, states, constants= None, training= None, ** kwargs):
            //    # Recover per-cell states.
            //    state_size = (self.state_size[::- 1]
            //                  if self.reverse_state_order else self.state_size)
            //    nested_states = nest.pack_sequence_as(state_size, nest.flatten(states))

            //    # Call the cells in order and store the returned states.
            //    new_nested_states = []
            //    for cell, states in zip(self.cells, nested_states) :
            //      states = states if nest.is_nested(states) else [states]
            //# TF cell does not wrap the state into list when there is only one state.
            //    is_tf_rnn_cell = getattr(cell, '_is_tf_rnn_cell', None) is not None
            //      states = states[0] if len(states) == 1 and is_tf_rnn_cell else states
            //      if generic_utils.has_arg(cell.call, 'training'):
            //        kwargs['training'] = training
            //      else:
            //        kwargs.pop('training', None)
            //      # Use the __call__ function for callable objects, eg layers, so that it
            //      # will have the proper name scopes for the ops, etc.
            //      cell_call_fn = cell.__call__ if callable(cell) else cell.call
            //      if generic_utils.has_arg(cell.call, 'constants'):
            //        inputs, states = cell_call_fn(inputs, states,
            //                                      constants= constants, ** kwargs)
            //      else:
            //        inputs, states = cell_call_fn(inputs, states, ** kwargs)
            //      new_nested_states.append(states)

            //    return inputs, nest.pack_sequence_as(state_size,
            //                                         nest.flatten(new_nested_states))
        }

        public void build()
        {
            throw new NotImplementedException();
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

        public override LayerArgs get_config()
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
    }
}
