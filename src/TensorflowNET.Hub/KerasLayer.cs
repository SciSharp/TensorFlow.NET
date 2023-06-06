using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Common.Types;
using Tensorflow.Keras.Engine;
using Tensorflow.Train;
using Tensorflow.Training;
using Tensorflow.Training.Saving.SavedModel;
using static Tensorflow.Binding;

namespace Tensorflow.Hub
{
    public class KerasLayer : Layer
    {
        private string _handle;
        private LoadOptions? _load_options;
        private Trackable _func;
        private Func<Tensors, Tensors> _callable;

        public KerasLayer(string handle, bool trainable = false, LoadOptions? load_options = null) :
            base(new Keras.ArgsDefinition.LayerArgs() { Trainable = trainable })
        {
            _handle = handle;
            _load_options = load_options;

            _func = load_module(_handle, _load_options);
            _track_trackable(_func, "_func");
            // TODO(Rinne): deal with _is_hub_module_v1.

            _callable = _get_callable();
            _setup_layer(trainable);
        }

        private void _setup_layer(bool trainable = false)
        {
            HashSet<string> trainable_variables;
            if (_func is Layer layer)
            {
                foreach (var v in layer.TrainableVariables)
                {
                    _add_existing_weight(v, true);
                }
                trainable_variables = new HashSet<string>(layer.TrainableVariables.Select(v => v.UniqueId));
            }
            else if (_func.CustomizedFields.TryGetValue("trainable_variables", out var obj) && obj is IEnumerable<Trackable> trackables)
            {
                foreach (var trackable in trackables)
                {
                    if (trackable is IVariableV1 v)
                    {
                        _add_existing_weight(v, true);
                    }
                }
                trainable_variables = new HashSet<string>(trackables.Where(t => t is IVariableV1).Select(t => (t as IVariableV1).UniqueId));
            }
            else
            {
                trainable_variables = new HashSet<string>();
            }

            if (_func is Layer)
            {
                layer = (Layer)_func;
                foreach (var v in layer.Variables)
                {
                    if (!trainable_variables.Contains(v.UniqueId))
                    {
                        _add_existing_weight(v, false);
                    }
                }
            }
            else if (_func.CustomizedFields.TryGetValue("variables", out var obj) && obj is IEnumerable<Trackable> total_trackables)
            {
                foreach (var trackable in total_trackables)
                {
                    if (trackable is IVariableV1 v && !trainable_variables.Contains(v.UniqueId))
                    {
                        _add_existing_weight(v, false);
                    }
                }
            }

            if (_func.CustomizedFields.ContainsKey("regularization_losses"))
            {
                if ((_func.CustomizedFields["regularization_losses"] as ListWrapper)?.Count > 0)
                {
                    throw new NotImplementedException("The regularization_losses loading has not been supported yet, " +
                        "please submit an issue to https://github.com/SciSharp/TensorFlow.NET/issues to let us know and add a feature.");
                }
            }
        }

        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optionalArgs = null)
        {
            _check_trainability();

            // TODO(Rinne): deal with training_argument

            var result = _callable(inputs);

            return _apply_output_shape_if_set(inputs, result);
        }

        private void _check_trainability()
        {
            if (!Trainable) return;

            // TODO(Rinne): deal with _is_hub_module_v1 and signature

            if (TrainableWeights is null || TrainableWeights.Count == 0)
            {
                tf.Logger.Error("hub.KerasLayer is trainable but has zero trainable weights.");
            }
        }

        private Tensors _apply_output_shape_if_set(Tensors inputs, Tensors result)
        {
            // TODO(Rinne): implement it.
            return result;
        }

        private void _add_existing_weight(IVariableV1 weight, bool? trainable = null)
        {
            bool is_trainable;
            if (trainable is null)
            {
                is_trainable = weight.Trainable;
            }
            else
            {
                is_trainable = trainable.Value;
            }
            add_weight(weight.Name, weight.shape, weight.dtype, trainable: is_trainable, getter: x => weight);
        }

        private Func<Tensors, Tensors> _get_callable()
        {
            if (_func is Layer layer)
            {
                return x => layer.Apply(x);
            }
            if (_func.CustomizedFields.ContainsKey("__call__"))
            {
                if (_func.CustomizedFields["__call__"] is RestoredFunction function)
                {
                    return x => function.Apply(x);
                }
            }
            throw new ValueError("Cannot get the callable from the model.");
        }

        private static Trackable load_module(string handle, LoadOptions? load_options = null)
        {
            //var set_load_options = load_options ?? LoadContext.get_load_option();
            return module_v2.load(handle, load_options);
        }
    }
}
