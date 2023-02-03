using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Google.Protobuf;
using ICSharpCode.SharpZipLib.Zip;
using Tensorflow.Checkpoint;
using Tensorflow.Contexts;
using Tensorflow.Functions;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Utils;
using Tensorflow.ModelSaving;
using Tensorflow.Train;
using Tensorflow.Exceptions;
using Tensorflow.IO;
using Tensorflow.Keras.Optimizers;
using ThirdParty.Tensorflow.Python.Keras.Protobuf;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Saving.SavedModel;

public partial class KerasSavedModelUtils
{
    public static void Save(Model model, string filepath, bool overwrite, bool include_optimizer, ConcreteFunction? signatures,
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

        OptimizerV2? orig_optimizer = null;
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
                Metadata = layer.TrackingMetadata
            };

            metadata.Nodes.Add(saved_object);
        }

        return metadata;
    }

    
}