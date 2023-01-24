using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers.Rnn;
using Tensorflow.Keras.Metrics;
using Tensorflow.Train;

namespace Tensorflow.Keras.Saving.SavedModel
{
    // TODO: revise the name of these "Attributes". Since "Attribute" is a significant feature of C#, 
    // Using the name "Attributes" may be quite confusing.
    /// <summary>
    /// Class that tracks and validates all serialization attributes.
    /// </summary>
    public abstract class SerializedAttributes
    {
        protected IDictionary<string, Trackable?> _object_dict;
        protected IDictionary<string, Function?> _function_dict;
        protected AutoTrackable _keras_trackable;
        protected HashSet<string> _all_functions;
        protected HashSet<string> _all_checkpointable_objects;

        protected SerializedAttributes()
        {
            _object_dict= new Dictionary<string, Trackable?>();
            _function_dict= new Dictionary<string, Function?>();
            _keras_trackable= new AutoTrackable();
            _all_functions= new HashSet<string>();
            _all_checkpointable_objects= new HashSet<string>();
        }

        protected SerializedAttributes(IEnumerable<string> checkpointable_objects, IEnumerable<string> functions)
        {
            _object_dict = new Dictionary<string, Trackable?>();
            _function_dict = new Dictionary<string, Function?>();
            _keras_trackable = new AutoTrackable();

            _all_checkpointable_objects = new HashSet<string>(checkpointable_objects);
            _all_functions = new HashSet<string>(functions);
        }

        public IDictionary<string, Function> Functions => _function_dict.TakeWhile(x => x.Value is not null).ToDictionary(x => x.Key, x => x.Value!);

        public IDictionary<string, Trackable> CheckpointableObjects => _object_dict.TakeWhile(x => x.Value is not null).ToDictionary(x => x.Key, x => x.Value!);

        /// <summary>
        /// Returns functions to attach to the root object during serialization.
        /// </summary>
        public IDictionary<string, Function> FunctionsToSerialize
        {
            get
            {
                Dictionary<string, Function> functions = new();
                foreach(var pair in Functions)
                {
                    if (_all_functions.Contains(pair.Key))
                    {
                        // TODO: deal with `LayerCall`.
                        functions[pair.Key] = pair.Value;
                    }
                }
                return functions;
            }
        }

        /// <summary>
        /// Returns objects to attach to the root object during serialization.
        /// </summary>
        public IDictionary<string, Trackable> ObjectsToSerialize
        {
            get
            {
                var objects = CheckpointableObjects.TakeWhile( x=> _all_checkpointable_objects.Contains(x.Key)).ToDictionary(x => x.Key, x => x.Value);
                objects[Constants.KERAS_ATTR] = _keras_trackable;
                return objects;
            }
        }

        /// <summary>
        /// Saves function dictionary, and validates dictionary values.
        /// </summary>
        /// <param name="function_dict"></param>
        public IDictionary<string, Function> set_and_validate_functions(IDictionary<string, Function> function_dict)
        {
            foreach(var key in _all_functions)
            {
                if (function_dict.ContainsKey(key))
                {
                    // TODO: deal with type `LayerCall`.
                    var fn = function_dict[key];
                    if (fn is not null && (fn is not Function))
                    {
                        throw new ValueError($"Function dictionary contained a non-function object: {function_dict[key]} (for key {key}).");
                    }
                    _function_dict[key] = fn;

                    var tf_fn = fn; // TODO: deal with type `LayerCall`.

                    // Warning: this implmentation should be considered again.
                    var properties = _keras_trackable.GetType().GetProperties();
                    foreach (var property in properties)
                    {
                        if(property.Name == key)
                        {
                            property.SetValue(_keras_trackable, tf_fn);
                            break;
                        }
                    }
                }
                else
                {
                    throw new ValueError($"Function {key} missing from serialized function dict.");
                }
            }
            return Functions;
        }

        /// <summary>
        /// Saves objects to a dictionary, and validates the values.
        /// </summary>
        /// <param name="object_dict"></param>
        public IDictionary<string, Trackable> set_and_validate_objects(IDictionary<string, Trackable> object_dict)
        {
            foreach(var key in _all_checkpointable_objects)
            {
                if (object_dict.ContainsKey(key))
                {
                    _object_dict[key] = object_dict[key];
                    // Warning: this implmentation should be considered again.
                    var properties = _keras_trackable.GetType().GetProperties();
                    foreach (var property in properties)
                    {
                        if (property.Name == key)
                        {
                            property.SetValue(_keras_trackable, object_dict[key]);
                            break;
                        }
                    }
                }
                else
                {
                    throw new ValueError($"Object {key} missing from serialized object dict.");
                }
            }
            return CheckpointableObjects;
        }

        /// <summary>
        /// Returns a new SerializedAttribute object (corresponding to `new` of tensorflow python).
        /// </summary>
        /// <returns></returns>
        public static SerializedAttributes Create(Trackable obj)
        {
            if(obj is Model)
            {
                return new ModelAttributes();
            }
            else if(obj is Metric)
            {
                return new MetricAttributes();
            }
            else if(obj is RNN)
            {
                return new RNNAttributes();
            }
            else if(obj is Layer)
            {
                return new LayerAttributes();
            }
            else
            {
                throw new TypeError($"Internal error during serialization: Expected Keras Layer object, got {obj} of type {obj.GetType()}");
            }
        }

        protected virtual (IEnumerable<string>, IEnumerable<string>) get_objects_and_functions_recursively(IEnumerable<string>? checkpointable_objects, IEnumerable<string>? functions)
        {
            return (checkpointable_objects ?? (new List<string>()), functions ?? (new List<string>()));
        }
    }

    // Note that the current implementation still has some potential risks.
    // The tensorflow python says that this class is "Common endpoints shared by all models loadable by Keras".
    // However, currently it's just a normal class.
    public class CommonEndPoints: SerializedAttributes
    {
        protected override (IEnumerable<string>, IEnumerable<string>) get_objects_and_functions_recursively(IEnumerable<string>? checkpointable_objects, IEnumerable<string>? functions)
        {
            if(checkpointable_objects is null)
            {
                checkpointable_objects = new List<string>();
            }
            if(functions is null)
            {
                functions = new List<string>();
            }
            return base.get_objects_and_functions_recursively(
                checkpointable_objects.Concat(new string[] { "variables", "trainable_variables", "regularization_losses" }),
                // TODO: remove the `__call__`.
                functions.Concat(new string[] {"__call__", "call_and_return_all_conditional_losses", "_default_save_signature" })
            );
        }
    }

    public class LayerAttributes: CommonEndPoints
    {
        protected override (IEnumerable<string>, IEnumerable<string>) get_objects_and_functions_recursively(IEnumerable<string>? checkpointable_objects, IEnumerable<string>? functions)
        {
            if (checkpointable_objects is null)
            {
                checkpointable_objects = new List<string>();
            }
            if (functions is null)
            {
                functions = new List<string>();
            }
            return base.get_objects_and_functions_recursively(
                checkpointable_objects.Concat(new string[] { "non_trainable_variables", "layers", "metrics", "layer_regularization_losses", "layer_metrics" }),
                functions.Concat(new string[] { "call_and_return_conditional_losses", "activity_regularizer_fn" })
            );
        }
    }

    public class ModelAttributes: LayerAttributes
    {
        protected override (IEnumerable<string>, IEnumerable<string>) get_objects_and_functions_recursively(IEnumerable<string>? checkpointable_objects, IEnumerable<string>? functions)
        {
            if (checkpointable_objects is null)
            {
                checkpointable_objects = new List<string>();
            }
            if (functions is null)
            {
                functions = new List<string>();
            }
            return base.get_objects_and_functions_recursively(checkpointable_objects,functions);
        }
    }

    public class MetricAttributes : SerializedAttributes
    {
        protected override (IEnumerable<string>, IEnumerable<string>) get_objects_and_functions_recursively(IEnumerable<string>? checkpointable_objects, IEnumerable<string>? functions)
        {
            if (checkpointable_objects is null)
            {
                checkpointable_objects = new List<string>();
            }
            if (functions is null)
            {
                functions = new List<string>();
            }
            return base.get_objects_and_functions_recursively(
                checkpointable_objects.Concat(new string[] { "variables" }),
                functions
            );
        }
    }

    public class RNNAttributes: LayerAttributes
    {
        protected override (IEnumerable<string>, IEnumerable<string>) get_objects_and_functions_recursively(IEnumerable<string>? checkpointable_objects, IEnumerable<string>? functions)
        {
            if (checkpointable_objects is null)
            {
                checkpointable_objects = new List<string>();
            }
            if (functions is null)
            {
                functions = new List<string>();
            }
            return base.get_objects_and_functions_recursively(
                checkpointable_objects.Concat(new string[] { "states" }),
                functions
            );
        }
    }
}
