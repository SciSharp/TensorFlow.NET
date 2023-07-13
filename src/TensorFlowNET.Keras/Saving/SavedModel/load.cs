using System.IO;
using Tensorflow.Train;
using ThirdParty.Tensorflow.Python.Keras.Protobuf;

namespace Tensorflow.Keras.Saving.SavedModel;

public class KerasLoadModelUtils
{
    /// <summary>
    /// Corresponding to keras/saving/save.py/load_model
    /// </summary>
    /// <param name="filepath"></param>
    /// <param name="custom_objects"></param>
    /// <param name="compile"></param>
    /// <param name="options"></param>
    /// <returns></returns>
    public static Trackable load_model(string filepath, IDictionary<string, object>? custom_objects = null,
        bool compile = true, LoadOptions? options = null)
    {
        using var savingScope = SharedObjectSavingScope.Enter();

        using var ctx = LoadContext.load_context(options);

        if (!File.Exists(filepath) && !Directory.Exists(filepath))
        {
            throw new IOException($"No file or directory found at {filepath}.");
        }

        if (Directory.Exists(filepath))
        {
            return load(filepath, compile, options);
        }
        else
        {
            throw new NotImplementedException("Model load of h5 format has not been supported. Please submit an issue to https://github.com/SciSharp/TensorFlow.NET/issues if it's needed.");
        }
    }

    private static Trackable load(string path, bool compile = true, LoadOptions? options = null)
    {
        SavedMetadata metadata;
        var meta_graph_def = Loader.parse_saved_model(path).MetaGraphs[0];
        var object_graph_def = meta_graph_def.ObjectGraphDef;
        string path_to_metadata_pb = Path.Combine(path, Constants.SAVED_METADATA_PATH);
        if (File.Exists(path_to_metadata_pb))
        {
            using var stream = new FileStream(path_to_metadata_pb, FileMode.Open, FileAccess.Read);
            metadata = SavedMetadata.Parser.ParseFrom(stream);
        }
        else
        {
            throw new NotImplementedException("SavedModel saved prior to TF 2.5 detected when loading Keras model, please" +
                " use higher version or submit an issue to https://github.com/SciSharp/TensorFlow.NET/issues. to let us know you need it.");
        }

        if (metadata.Nodes is null || metadata.Nodes.Count == 0)
        {
            return Loader.load(path, options: options) as Model;
        }

        var keras_loader = new KerasObjectLoader(metadata, object_graph_def);
        keras_loader.load_layers(compile: compile);

        Dictionary<string, (Trackable, Action<object, object, object>)> nodes_to_load = new();
        nodes_to_load["root"] = (null, null);
        foreach(var item in keras_loader.LoadedNodes)
        {
            nodes_to_load[keras_loader.get_path(item.Key)] = item.Value;
        }
        var loaded = Loader.load_partial(path, nodes_to_load, options);

        keras_loader.finalize_objects();
        keras_loader.del_tracking();

        var model = loaded["root"];

        if (model is Model && compile)
        {
            // TODO(Rinne): implement it.
        }

        if (!tf.Context.executing_eagerly())
        {
            // TODO(Rinne): implement it.
        }

        return model;
    }
}
