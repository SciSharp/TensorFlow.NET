using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Google.Protobuf;
using Tensorflow.Functions;
using Tensorflow.Keras.Engine;
using Tensorflow.ModelSaving;
using Tensorflow.Train;
using Tensorflow.Keras.Optimizers;
using ThirdParty.Tensorflow.Python.Keras.Protobuf;
using static Tensorflow.Binding;
using Tensorflow.Training;
using System.Diagnostics;

namespace Tensorflow.Keras.Saving.SavedModel;

public partial class KerasSavedModelUtils
{
    public static void save_model(Model model, string filepath, bool overwrite, bool include_optimizer, ConcreteFunction? signatures,
        SaveOptions? options, bool save_traces = true)
    {
        if (!overwrite && File.Exists(filepath))
        {
            throw new Exception("The file already exists but is not allowed to overwrite it.");
        }

        if (save_traces)
        {
            if(should_skip_serialization(model))
            {
                throw new NotImplementedException();
            }
        }

        IOptimizer? orig_optimizer = null;
        if (!include_optimizer)
        {
            orig_optimizer = model.Optimizer;
            model.Optimizer = null;
            model._delete_tracking("optimizer");
        }

        IList<Trackable> saved_nodes;
        IDictionary<Trackable, IEnumerable<TrackableReference>> node_paths;
        // skip two scopes of python
        using (KerasSavedModelUtils.keras_option_scope(save_traces))
        {
            (saved_nodes, node_paths) = Tensorflow.SavedModelUtils.save_and_return_nodes(model, filepath, signatures, options);
        }

        var metadata = generate_keras_metadata(saved_nodes, node_paths);
        File.WriteAllBytes(Path.Combine(filepath, Constants.SAVED_METADATA_PATH), metadata.ToByteArray());
        //File.WriteAllText(Path.Combine(filepath, Constants.SAVED_METADATA_PATH), metadata.ToString());

        if (!include_optimizer)
        {
            model.Optimizer = orig_optimizer!;
        }
    }

    public static SavedMetadata generate_keras_metadata(IList<Trackable> saved_nodes,
        IDictionary<Trackable, IEnumerable<TrackableReference>> node_paths)
    {
        var metadata = new SavedMetadata();
        for (int i = 0; i < saved_nodes.Count; i++)
        {
            var node = saved_nodes[i];
            if (node is not Layer)
            {
                continue;
            }

            Layer layer = (Layer)node;

            var path = node_paths[node];
            string node_path;
            if (path is null || path.Count() == 0)
            {
                node_path = "root";
            }
            else
            {
                node_path = $"root.{string.Join(".", path.Select(x => x.Name))}";
            }
            
            ThirdParty.Tensorflow.Python.Keras.Protobuf.SavedObject saved_object = new()
            {
                NodeId = i,
                NodePath = node_path,
                Version = new ThirdParty.Tensorflow.Python.Keras.Protobuf.VersionDef()
                {
                    Producer = 2,
                    MinConsumer = 1,
                    BadConsumers = {  }
                },
                Identifier = layer.ObjectIdentifier,
                Metadata = layer.GetTrackingMetadata()
            };

            metadata.Nodes.Add(saved_object);
        }

        return metadata;
    }

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
        }).ToArray());
        var trainable_variables = TrackableDataStructure.wrap_or_unwrap(layer.TrainableVariables.Select(x =>
        {
            if (x is ResourceVariable or RefVariable) return (Trackable)x;
            else throw new TypeError($"The type{x.GetType()} is not supported for the wrapping of layer.");
        }).ToArray());
        var non_trainable_variables = TrackableDataStructure.wrap_or_unwrap(layer.NonTrainableVariables.Select(x =>
        {
            if (x is ResourceVariable or RefVariable) return (Trackable)x;
            else throw new TypeError($"The type{x.GetType()} is not supported for the wrapping of layer.");
        }).ToArray());
        var layers = TrackableDataStructure.wrap_or_unwrap(list_all_layers(layer).Select(x => x.GetTrackable()).ToArray());

        Dictionary<string, Trackable> res = new();
        Debug.Assert(variables is Trackable);
        Debug.Assert(trainable_variables is Trackable);
        Debug.Assert(non_trainable_variables is Trackable);
        Debug.Assert(layers is Trackable);
        res["variables"] = variables as Trackable;
        res["trainable_variables"] = trainable_variables as Trackable;
        res["non_trainable_variables"] = non_trainable_variables as Trackable;
        res["layers"] = layers as Trackable;

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

        // high priority
        // TODO: deal with type `RevivedLayer` and `Sequential`.

        // skip the process because of lack of APIs of `Layer`.

        return new Dictionary<string, Trackable>();
    }
}
