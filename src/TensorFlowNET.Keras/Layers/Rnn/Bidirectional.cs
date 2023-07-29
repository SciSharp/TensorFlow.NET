using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Common.Types;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Saving;

namespace Tensorflow.Keras.Layers
{
    /// <summary>
    /// Bidirectional wrapper for RNNs.
    /// </summary>
    public class Bidirectional: Wrapper
    {
        int _num_constants = 0;
        bool _support_masking = true;
        bool _return_state;
        bool _stateful;
        bool _return_sequences;
        BidirectionalArgs _args;
        RNNArgs _layer_args_copy;
        RNN _forward_layer;
        RNN _backward_layer;
        RNN _layer;
        InputSpec _input_spec;
        public Bidirectional(BidirectionalArgs args):base(args)
        {
            _args = args;
            if (_args.Layer is not ILayer)
                throw new ValueError(
                "Please initialize `Bidirectional` layer with a " +
                $"`tf.keras.layers.Layer` instance. Received: {_args.Layer}");

            if (_args.BackwardLayer is not null && _args.BackwardLayer is not ILayer)
                throw new ValueError(
                "`backward_layer` need to be a `tf.keras.layers.Layer` " +
                $"instance. Received: {_args.BackwardLayer}");
            if (!new List<string> { "sum", "mul", "ave", "concat", null }.Contains(_args.MergeMode))
            {
                throw new ValueError(
                    $"Invalid merge mode. Received: {_args.MergeMode}. " +
                    "Merge mode should be one of " +
                    "{\"sum\", \"mul\", \"ave\", \"concat\", null}"
                );
            }
            if (_args.Layer is RNN)
            {
                _layer = _args.Layer as RNN;
            }
            else
            {
                throw new ValueError(
                    "Bidirectional only support RNN instance such as LSTM or GRU");
            }
            _return_state = _layer.Args.ReturnState;
            _return_sequences = _layer.Args.ReturnSequences;
            _stateful = _layer.Args.Stateful;
            _layer_args_copy = _layer.Args.Clone();
            // We don't want to track `layer` since we're already tracking the two
            // copies of it we actually run.
            // TODO(Wanglongzhi2001), since the feature of setattr_tracking has not been implemented.
            // _setattr_tracking = false;
            // super().__init__(layer, **kwargs)
            // _setattr_tracking = true;

            // Recreate the forward layer from the original layer config, so that it
            // will not carry over any state from the layer.
            if (_layer is LSTM)
            {
                var arg = _layer_args_copy as LSTMArgs;
                _forward_layer = new LSTM(arg);
            }
            else if(_layer is SimpleRNN)
            {
                var arg = _layer_args_copy as SimpleRNNArgs;
                _forward_layer = new SimpleRNN(arg);
            }
            // TODO(Wanglongzhi2001), add GRU if case.
            else
            {
                _forward_layer = new RNN(_layer.Cell, _layer_args_copy);
            }
            //_forward_layer = _recreate_layer_from_config(_layer);
            if (_args.BackwardLayer is null)
            {
                _backward_layer = _recreate_layer_from_config(_layer, go_backwards:true);
            }
            else
            {
                _backward_layer = _args.BackwardLayer as RNN;
            }
            _forward_layer.Name = "forward_" + _forward_layer.Name;
            _backward_layer.Name = "backward_" + _backward_layer.Name;
            _verify_layer_config();

            void force_zero_output_for_mask(RNN layer)
            {
                layer.Args.ZeroOutputForMask = layer.Args.ReturnSequences;
            }

            force_zero_output_for_mask(_forward_layer);
            force_zero_output_for_mask(_backward_layer);

            if (_args.Weights is not null)
            {
                var nw = len(_args.Weights);
                _forward_layer.set_weights(_args.Weights[$":,{nw / 2}"]);
                _backward_layer.set_weights(_args.Weights[$"{nw / 2},:"]);
            }

            _input_spec = _layer.InputSpec;
        }

        private void _verify_layer_config()
        {
            if (_forward_layer.Args.GoBackwards == _backward_layer.Args.GoBackwards)
            {
                throw new ValueError(
                    "Forward layer and backward layer should have different " +
                    "`go_backwards` value." +
                    "forward_layer.go_backwards = " +
                    $"{_forward_layer.Args.GoBackwards}," +
                    "backward_layer.go_backwards = " +
                    $"{_backward_layer.Args.GoBackwards}");
            }
            if (_forward_layer.Args.Stateful != _backward_layer.Args.Stateful)
            {
                throw new ValueError(
                    "Forward layer and backward layer are expected to have "+
                    $"the same value for attribute stateful, got "+
                    $"{_forward_layer.Args.Stateful} for forward layer and "+
                    $"{_backward_layer.Args.Stateful} for backward layer");
            }
            if (_forward_layer.Args.ReturnState != _backward_layer.Args.ReturnState)
            {
                throw new ValueError(
                    "Forward layer and backward layer are expected to have " +
                    $"the same value for attribute return_state, got " +
                    $"{_forward_layer.Args.ReturnState} for forward layer and " +
                    $"{_backward_layer.Args.ReturnState} for backward layer");
            }
            if (_forward_layer.Args.ReturnSequences != _backward_layer.Args.ReturnSequences)
            {
                throw new ValueError(
                    "Forward layer and backward layer are expected to have " +
                    $"the same value for attribute return_sequences, got " +
                    $"{_forward_layer.Args.ReturnSequences} for forward layer and " +
                    $"{_backward_layer.Args.ReturnSequences} for backward layer");
            }
        }

        private RNN _recreate_layer_from_config(RNN layer, bool go_backwards = false)
        {
            var config = layer.get_config() as RNNArgs;
            var cell = layer.Cell;
            if (go_backwards)
            {
                config.GoBackwards = !config.GoBackwards;
            }

            if (layer is LSTM)
            {
                var arg = config as LSTMArgs;
                return new LSTM(arg);
            }
            else if(layer is SimpleRNN)
            {
                var arg = config as SimpleRNNArgs;
                return new SimpleRNN(arg);
            }
            // TODO(Wanglongzhi2001), add GRU if case.
            else
            {
                return new RNN(cell, config);
            }
        }

        public override void build(KerasShapesWrapper input_shape)
        {
            _buildInputShape = input_shape;
            tf_with(ops.name_scope(_forward_layer.Name), scope=>
            {
                _forward_layer.build(input_shape);
            });
            tf_with(ops.name_scope(_backward_layer.Name), scope =>
            {
                _backward_layer.build(input_shape);
            });
            built = true;
        }

        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            // `Bidirectional.call` implements the same API as the wrapped `RNN`.
            Tensors forward_inputs;
            Tensors backward_inputs;
            Tensors forward_state;
            Tensors backward_state;
            // if isinstance(inputs, list) and len(inputs) > 1:
            if (inputs.Length > 1)
            {
                // initial_states are keras tensors, which means they are passed
                // in together with inputs as list. The initial_states need to be
                // split into forward and backward section, and be feed to layers
                // accordingly.
                forward_inputs = new Tensors { inputs[0] };
                backward_inputs = new Tensors { inputs[0] };
                var pivot = (len(inputs) - _num_constants) / 2 + 1;
                // add forward initial state
                forward_inputs.Concat(new Tensors { inputs[$"1:{pivot}"] });
                if (_num_constants != 0)
                    // add backward initial state
                    backward_inputs.Concat(new Tensors { inputs[$"{pivot}:"] });
                else
                {
                    // add backward initial state
                    backward_inputs.Concat(new Tensors { inputs[$"{pivot}:{-_num_constants}"] });
                    // add constants for forward and backward layers
                    forward_inputs.Concat(new Tensors { inputs[$"{-_num_constants}:"] });
                    backward_inputs.Concat(new Tensors { inputs[$"{-_num_constants}:"] });
                }
                forward_state = null;
                backward_state = null;
            }
            else if (state is not null)
            {
                // initial_states are not keras tensors, eg eager tensor from np
                // array.  They are only passed in from kwarg initial_state, and
                // should be passed to forward/backward layer via kwarg
                // initial_state as well.
                forward_inputs = inputs;
                backward_inputs = inputs;
                var half = len(state) / 2;
                forward_state = state[$":{half}"];
                backward_state = state[$"{half}:"];
            }
            else
            {
                forward_inputs = inputs;
                backward_inputs = inputs;
                forward_state = null;
                backward_state = null;
            }
            var y = _forward_layer.Apply(forward_inputs, forward_state);
            var y_rev = _backward_layer.Apply(backward_inputs, backward_state);

            Tensors states = new();
            if (_return_state)
            {
                states = y["1:"] + y_rev["1:"];
                y = y[0];
                y_rev = y_rev[0];
            }

            if (_return_sequences)
            {
                int time_dim = _forward_layer.Args.TimeMajor ? 0 : 1;
                y_rev = keras.backend.reverse(y_rev, time_dim);
            }
            Tensors output;
            if (_args.MergeMode == "concat")
                output = keras.backend.concatenate(new Tensors { y.Single(), y_rev.Single() });
            else if (_args.MergeMode == "sum")
                output = y.Single() + y_rev.Single();
            else if (_args.MergeMode == "ave")
                output = (y.Single() + y_rev.Single()) / 2;
            else if (_args.MergeMode == "mul")
                output = y.Single() * y_rev.Single();
            else if (_args.MergeMode is null)
                output = new Tensors { y.Single(), y_rev.Single() };
            else
                throw new ValueError(
                        "Unrecognized value for `merge_mode`. " +
                        $"Received: {_args.MergeMode}" +
                        "Expected values are [\"concat\", \"sum\", \"ave\", \"mul\"]");
            if (_return_state)
            {
                if (_args.MergeMode is not null)
                    return new Tensors { output.Single(), states.Single()};
            }
            return output;
        }
    }
}
