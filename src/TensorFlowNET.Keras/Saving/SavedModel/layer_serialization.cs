using System.Collections.Generic;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Utils;
using Tensorflow.Train;

namespace Tensorflow.Keras.Saving.SavedModel;

public class LayerSavedModelSaver: SavedModelSaver
{
    private Layer _layer;
    public LayerSavedModelSaver(Layer obj): base(obj)
    {
        _obj = obj;
        _layer = obj;
    }
    public override string ObjectIdentifier
    {
        get => Constants.LAYER_IDENTIFIER;
    }

    public override IDictionary<string, Trackable> objects_to_serialize(IDictionary<string, IDictionary<Trackable, ISerializedAttributes>> serialization_cache)
    {
        return get_serialized_attributes(serialization_cache).ObjectsToSerialize;
    }

    public override IDictionary<string, Trackable> functions_to_serialize(IDictionary<string, IDictionary<Trackable, ISerializedAttributes>> serialization_cache)
    {
        return get_serialized_attributes(serialization_cache).FunctionsToSerialize;
    }

    /// <summary>
    /// Generates or retrieves serialized attributes from cache.
    /// </summary>
    /// <param name="serialization_cache"></param>
    protected ISerializedAttributes get_serialized_attributes(IDictionary<string, IDictionary<Trackable, ISerializedAttributes>> serialization_cache)
    {
        // TODO: deal with cache.
        IDictionary<Trackable, ISerializedAttributes> keras_cache;
        if(serialization_cache is not null && serialization_cache.ContainsKey(Constants.KERAS_CACHE_KEY))
        {
            keras_cache = serialization_cache[Constants.KERAS_CACHE_KEY];
        }
        else
        {
            serialization_cache![Constants.KERAS_CACHE_KEY] = keras_cache = new Dictionary<Trackable, ISerializedAttributes>();
        }
        if (keras_cache.ContainsKey(_obj)) return keras_cache[_obj];

        var serialized_attr = keras_cache[_obj] = SerializedAttributes.Create(_obj);

        // TODO: complete the statement. Currently the `Layer` lacks member `_must_restore_from_config`.
        if (KerasSavedModelUtils.should_skip_serialization(_obj))
        {
            return serialized_attr;
        }

        var (object_dict, function_dict) = get_serialized_attributes_internal(serialization_cache);

        serialized_attr.set_and_validate_objects(object_dict);
        serialized_attr.set_and_validate_functions(function_dict);
        return serialized_attr;
    }

    /// <summary>
    /// Returns dictionary of serialized attributes.
    /// </summary>
    /// <param name="serialization_cache"></param>
    private (IDictionary<string, Trackable>, IDictionary<string, Trackable>) get_serialized_attributes_internal(IDictionary<string, IDictionary<Trackable, ISerializedAttributes>> serialization_cache)
    {
        var objects = KerasSavedModelUtils.wrap_layer_objects(_layer, serialization_cache);
        var functions = KerasSavedModelUtils.wrap_layer_functions(_layer, serialization_cache);

        functions["_default_save_signature"] = null;

        return (objects, functions);
    }

    public override string TrackingMetadata
    {
        get
        {
            JObject metadata = new JObject();
            metadata["name"] = _layer.Name;
            metadata["trainable"] = _layer.Trainable;
            // TODO: implement `expects_training_arg`.
            metadata["expects_training_arg"] = false;
            metadata["dtype"] = _layer.DType.as_python_name();
            metadata["batch_input_shape"] = _layer.BatchInputShape is null ? null : JToken.FromObject(_layer.BatchInputShape);
            // metadata["stateful"] = _obj.stateful;
            // metadata["must_restore_from_config"] = _obj.must_restore_from_config;
            // metadata["preserve_input_structure_in_config"] = _obj.preserve_input_structure_in_config;
            metadata["autocast"] = _layer.AutoCast;

            if(_layer.InputSpec is not null)
            {
                metadata["input_spec"] = generic_utils.serialize_keras_object(_layer.InputSpec);
            }

            metadata.Merge(get_serialized(_layer), new JsonMergeSettings
            {
                // Handle conflicts by using values from obj2
                MergeArrayHandling = MergeArrayHandling.Merge
            });
            // skip the check of `input_spec` and `build_input_shape` for the lack of members.
            // skip the check of `activity_regularizer` for the type problem.
            if(_layer.BuildInputShape is not null)
            {
                metadata["build_input_shape"] = JToken.FromObject(_layer.BuildInputShape);
            }
            return metadata.ToString();
        }
    }

    public static JObject get_serialized(Layer obj)
    {
        return generic_utils.serialize_keras_object(obj);
    }
}

public class InputLayerSavedModelSaver: SavedModelSaver
{
    public InputLayerSavedModelSaver(Layer obj) : base(obj)
    {
        
    }
    public override string ObjectIdentifier => Constants.INPUT_LAYER_IDENTIFIER;

    public override IDictionary<string, Trackable> functions_to_serialize(IDictionary<string, IDictionary<Trackable, ISerializedAttributes>> serialization_cache)
    {
        return new Dictionary<string, Trackable>();
    }

    public override IDictionary<string, Trackable> objects_to_serialize(IDictionary<string, IDictionary<Trackable, ISerializedAttributes>> serialization_cache)
    {
        return new Dictionary<string, Trackable>();
    }

    public override string TrackingMetadata
    {
        get
        {
            if(_obj is not InputLayer)
            {
                throw new TypeError($"The type {_obj.GetType()} cannot be recognized as an input layer.");
            }
            var layer = (InputLayer)_obj;
            var config = (layer.get_config() as InputLayerArgs)!;
            var info = new
            {
                class_name = layer.GetType().Name,
                name = layer.Name,
                dtype = layer.DType,
                sparse = config.Sparse,
                ragged = config.Ragged,
                batch_input_shape = layer.BatchInputShape,
                config = layer.get_config()
            };
            return JsonConvert.SerializeObject(info);
        }
    }
}
