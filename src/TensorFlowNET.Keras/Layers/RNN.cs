using System;
using System.Collections.Generic;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
// from tensorflow.python.distribute import distribution_strategy_context as ds_context;

namespace Tensorflow.Keras.Layers
{
    public class RNN : Layer
    {
        private RNNArgs args;
        private object input_spec = null; // or NoneValue??
        private object state_spec = null;
        private object _states = null;
        private object constants_spec = null;
        private int _num_constants = 0;

        public RNN(RNNArgs args) : base(PreConstruct(args))
        {
            this.args = args;
            SupportsMasking = true;

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

        private static RNNArgs PreConstruct(RNNArgs args)
        {
            if (args.Kwargs == null)
            {
                args.Kwargs = new Dictionary<string, object>();
            }

            // If true, the output for masked timestep will be zeros, whereas in the
            // false case, output from previous timestep is returned for masked timestep.
            var zeroOutputForMask = (bool)args.Kwargs.Get("zero_output_for_mask", false);

            TensorShape input_shape;
            var propIS = (TensorShape)args.Kwargs.Get("input_shape", null);
            var propID = (int?)args.Kwargs.Get("input_dim", null);
            var propIL = (int?)args.Kwargs.Get("input_length", null);

            if (propIS == null && (propID != null || propIL != null))
            {
                input_shape = new TensorShape(
                    propIL ?? -1,
                    propID ?? -1);
                args.Kwargs["input_shape"] = input_shape;
            }

            return args;
        }

        public RNN New(LayerRnnCell cell,
            bool return_sequences = false,
            bool return_state = false,
            bool go_backwards = false,
            bool stateful = false,
            bool unroll = false,
            bool time_major = false)
                => new RNN(new RNNArgs
                {
                    Cell = cell,
                    ReturnSequences = return_sequences,
                    ReturnState = return_state,
                    GoBackwards = go_backwards,
                    Stateful = stateful,
                    Unroll = unroll,
                    TimeMajor = time_major
                });

        public RNN New(IList<RnnCell> cell,
            bool return_sequences = false,
            bool return_state = false,
            bool go_backwards = false,
            bool stateful = false,
            bool unroll = false,
            bool time_major = false)
                => new RNN(new RNNArgs
                {
                    Cell = new StackedRNNCells(new StackedRNNCellsArgs { Cells = cell }),
                    ReturnSequences = return_sequences,
                    ReturnState = return_state,
                    GoBackwards = go_backwards,
                    Stateful = stateful,
                    Unroll = unroll,
                    TimeMajor = time_major
                });


        protected Tensor get_initial_state(Tensor inputs)
        {
            return _generate_zero_filled_state_for_cell(null, null);
        }

        Tensor _generate_zero_filled_state_for_cell(LSTMCell cell, Tensor batch_size)
        {
            throw new NotImplementedException("");
        }

        // Check whether the state_size contains multiple states.
        public static bool _is_multiple_state(object state_size)
        {
            var myIndexerProperty = state_size.GetType().GetProperty("Item");
            return myIndexerProperty != null
                && myIndexerProperty.GetIndexParameters().Length == 1
                && !(state_size.GetType() == typeof(TensorShape));
        }
    }
}
