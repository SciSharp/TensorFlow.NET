using System;
using System.ComponentModel;
using System.Linq;
using Tensorflow.Common.Extensions;
using Tensorflow.Common.Types;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Saving;
using Tensorflow.Keras.Utils;

namespace Tensorflow.Keras.Layers
{
    public class StackedRNNCells : Layer, IRnnCell
    {
        public IList<IRnnCell> Cells { get; set; }
        public bool _reverse_state_order;

        public StackedRNNCells(IEnumerable<IRnnCell> cells, StackedRNNCellsArgs args) : base(args)
        {
            Cells = cells.ToList(); 

            _reverse_state_order = args.ReverseStateOrder;

            if (_reverse_state_order)
            {
                throw new WarningException("reverse_state_order=True in StackedRNNCells will soon " +
                                           "be deprecated. Please update the code to work with the " +
                                           "natural order of states if you rely on the RNN states, " +
                                           "eg RNN(return_state=True).");
            }
        }

        public bool SupportOptionalArgs => false;

        public INestStructure<long> StateSize
        {
            get
            {
                if (_reverse_state_order)
                {
                    var state_sizes = Cells.Reverse().Select(cell => cell.StateSize);
                    return new Nest<long>(state_sizes);
                }
                else
                {
                    var state_sizes = Cells.Select(cell => cell.StateSize);
                    return new Nest<long>(state_sizes);
                }
            }
        }

        public INestStructure<long> OutputSize
        {
            get
            {
                var lastCell = Cells.Last();
                if(lastCell.OutputSize is not null)
                {
                    return lastCell.OutputSize;
                }
                else if (RnnUtils.is_multiple_state(lastCell.StateSize))
                {
                    return new NestNode<long>(lastCell.StateSize.Flatten().First());
                }
                else
                {
                    return lastCell.StateSize;
                }
            }
        }

        public Tensors GetInitialState(Tensors inputs = null, Tensor batch_size = null, TF_DataType dtype = TF_DataType.DtInvalid)
        {
            var cells = _reverse_state_order ? Cells.Reverse() : Cells;
            List<Tensor> initial_states = new List<Tensor>();
            foreach (var cell in cells)
            {
                initial_states.Add(cell.GetInitialState(inputs, batch_size, dtype));
            }
            return new Tensors(initial_states);
        }

        protected override Tensors Call(Tensors inputs, Tensors states = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            // Recover per-cell states.
            var state_size = _reverse_state_order ? new NestList<long>(StateSize.Flatten().Reverse()) : StateSize;
            var nested_states = Nest.PackSequenceAs(state_size, Nest.Flatten(states).ToArray());

            var new_nest_states = Nest<Tensor>.Empty;
            // Call the cells in order and store the returned states.
            foreach (var (cell, internal_states) in zip(Cells, nested_states))
            {
                RnnOptionalArgs? rnn_optional_args = optional_args as RnnOptionalArgs;
                Tensors? constants = rnn_optional_args?.Constants;

                Tensors new_states;
                (inputs, new_states) = cell.Apply(inputs, internal_states, optional_args: new RnnOptionalArgs() { Constants = constants });

                new_nest_states = new_nest_states.MergeWith(new_states);
            }
            return Tensors.FromNest((inputs, Nest.PackSequenceAs(state_size, Nest.Flatten(new_nest_states).ToArray())));
        }

        public override void build(KerasShapesWrapper input_shape)
        {
            var shape = input_shape.ToSingleShape();
            foreach(var cell in Cells)
            {
                if(cell is Layer layer && !layer.Built)
                {
                    // ignored the name scope.
                    layer.build(shape);
                    layer.Built = true;
                }
                INestStructure<long> output_dim;
                if(cell.OutputSize is not null)
                {
                    output_dim = cell.OutputSize;
                }
                else if (RnnUtils.is_multiple_state(cell.StateSize))
                {
                    output_dim = new NestNode<long>(cell.StateSize.Flatten().First());
                }
                else
                {
                    output_dim = cell.StateSize;
                }
                shape = new Shape(new long[] { shape.dims[0] }.Concat(output_dim.Flatten()).ToArray());
            }
            this.Built = true;
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
    }
}
