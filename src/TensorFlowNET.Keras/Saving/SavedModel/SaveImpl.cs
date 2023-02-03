using System.Collections.Generic;
using System.Linq;
using Tensorflow.Keras.Engine;
using Tensorflow.Train;
using Tensorflow.Training;

namespace Tensorflow.Keras.Saving.SavedModel;

public partial class KerasSavedModelUtils
{
    public static bool should_skip_serialization(object layer)
    {
        return false;
    }

    /// <summary>
    /// Returns extra trackable objects to attach to the serialized layer.
    /// </summary>
    /// <param name="layer"></param>
    /// <param name="serialization_cache"></param>
    /// <returns></returns>
    public static IDictionary<string, Trackable> wrap_layer_objects(Layer layer, IDictionary<string, IDictionary<Trackable, ISerializedAttributes>> serialization_cache)
    {
        // TODO: deal with losses and metrics. Currently, `Layer` lacks these two APIs.

        // TODO: change the inherits of `Variable` and revise the implmentation.
        var variables = TrackableDataStructure.wrap_or_unwrap(layer.Variables.Select(x =>
        {
            if (x is ResourceVariable or RefVariable) return (Trackable)x;
            else throw new TypeError($"The type{x.GetType()} is not supported for the wrapping of layer.");
        }));
        var trainable_variables = TrackableDataStructure.wrap_or_unwrap(layer.TrainableVariables.Select(x =>
        {
            if (x is ResourceVariable or RefVariable) return (Trackable)x;
            else throw new TypeError($"The type{x.GetType()} is not supported for the wrapping of layer.");
        }));
        var non_trainable_variables = TrackableDataStructure.wrap_or_unwrap(layer.non_trainable_variables.Select(x => 
        { 
            if (x is ResourceVariable or RefVariable) return (Trackable)x; 
            else throw new TypeError($"The type{x.GetType()} is not supported for the wrapping of layer."); 
        }));

        Dictionary<string, Trackable> res = new();
        res["variables"] = variables;
        res["trainable_variables"] = trainable_variables;
        res["non_trainable_variables"] = non_trainable_variables;
        res["layers"] = TrackableDataStructure.wrap_or_unwrap(KerasSavedModelUtils.list_all_layers(layer).Select(x => x.GetTrackable()));

        return res;
    }

    /// <summary>
    /// Returns dict of wrapped layer call function and losses in tf.functions.
    /// </summary>
    /// <param name="layer"></param>
    /// <param name="serialization_cache"></param>
    /// <returns></returns>
    public static IDictionary<string, Trackable> wrap_layer_functions(Layer layer, IDictionary<string, IDictionary<Trackable, ISerializedAttributes>> serialization_cache)
    {
        // TODO: deal with type `RevivedLayer` and `Sequential`.

        // skip the process because of lack of APIs of `Layer`.

        return new Dictionary<string, Trackable>();
    }
}