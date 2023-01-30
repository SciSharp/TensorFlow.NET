using System.Collections.Generic;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Utils;
using Tensorflow.Train;

namespace Tensorflow.Keras.Saving.SavedModel;

public class LayerSavedModelSaver: SavedModelSaver
{
    private Layer _obj;
    public LayerSavedModelSaver(Layer obj): base(obj)
    {
        _obj = obj;
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
        var objects = KerasSavedModelUtils.wrap_layer_objects(_obj, serialization_cache);
        var functions = KerasSavedModelUtils.wrap_layer_functions(_obj, serialization_cache);

        functions["_default_save_signature"] = null;

        return (objects, functions);
    }

    public override string TrackingMetadata
    {
        get
        {
            JObject metadata = new JObject();
            metadata["name"] = _obj.Name;
            metadata["trainable"] = _obj.Trainable;
            // metadata["expects_training_arg"] = _obj._expects_training_arg;
            // metadata["dtype"] = policy.serialize(_obj._dtype_policy)
            metadata["batch_input_shape"] = _obj.BatchInputShape is null ? null : JToken.FromObject(_obj.BatchInputShape);
            // metadata["stateful"] = _obj.stateful;
            // metadata["must_restore_from_config"] = _obj.must_restore_from_config;
            // metadata["preserve_input_structure_in_config"] = _obj.preserve_input_structure_in_config;
            metadata["autocast"] = _obj.AutoCast;
            
            metadata.Merge(JObject.FromObject(get_serialized(_obj)), new JsonMergeSettings
            {
                // Handle conflicts by using values from obj2
                MergeArrayHandling = MergeArrayHandling.Merge
            });
            // skip the check of `input_spec` and `build_input_shape` for the lack of members.
            // skip the check of `activity_regularizer` for the type problem.
            return metadata.ToString();
        }
    }

    public static IDictionary<string, object> get_serialized(Layer obj)
    {
        // TODO: complete the implmentation (need to revise `get_config`).
        return new Dictionary<string, object>();
        //return generic_utils.serialize_keras_object(obj);
    }
}