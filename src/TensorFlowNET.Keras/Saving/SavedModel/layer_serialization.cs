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

    public override IDictionary<string, CheckpointableBase> objects_to_serialize(IDictionary<string, object> serialization_cache)
    {
        throw new System.NotImplementedException();
    }

    public override IDictionary<string, Function> functions_to_serialize(IDictionary<string, object> serialization_cache)
    {
        throw new System.NotImplementedException();
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
            metadata["batch_input_shape"] = JToken.FromObject(_obj.BatchInputShape);
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

    public static LayerConfig get_serialized(Layer obj)
    {
        return generic_utils.serialize_keras_object(obj);
    }
}