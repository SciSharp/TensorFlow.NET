using Google.Protobuf;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using Tensorflow.Checkpoint;
using Tensorflow.Operations;
using Tensorflow.Train;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public partial class Loader
    {
        public static SavedModel parse_saved_model(string export_dir)
        {
            var path_to_pbtxt = tf.io.gfile.join(export_dir, Constants.SAVED_MODEL_FILENAME_PBTXT);
            var path_to_pb = tf.io.gfile.join(export_dir, Constants.SAVED_MODEL_FILENAME_PB);

            SavedModel saved_model = new SavedModel();
            if (File.Exists(path_to_pb))
            {
                byte[] file_content;
                using(var f = new FileStream(path_to_pb, FileMode.Open, FileAccess.Read))
                {
                    file_content = new byte[f.Length];
                    Debug.Assert(f.Length <= int.MaxValue);
                    f.Read(file_content, 0, (int)f.Length);
                }
                // TODO: change to stream mode.
                saved_model.MergeFrom(file_content);
                return saved_model;
            }
            else if (File.Exists(path_to_pbtxt))
            {
                throw new NotImplementedException();
            }
            else
            {
                throw new IOException($"SavedModel file does not exist at: {export_dir}{Path.PathSeparator}" +
                    $"{{{Constants.SAVED_MODEL_FILENAME_PBTXT}|{Constants.SAVED_MODEL_FILENAME_PB}}}");
            }
        }

        // TODO: revise the type of `tags`
        public static Trackable load(string export_dir, object? tags = null, LoadOptions? options = null)
        {
            return load_partial(export_dir, null, tags, options)["root"];
        }

        public static IDictionary<string, Trackable> load_partial(string export_dir, IDictionary<string, (Trackable, Action<object, object, object>)>? filters, object? tags = null, LoadOptions? options = null)
        {
            if (options is null)
            {
                options = new LoadOptions();
            }
            if (tags is not null)
            {
                throw new NotImplementedException();
            }
            var (saved_model_proto, debug_info) = Loader.parse_saved_model_with_debug_info(export_dir);

            Trackable root = null;
            Loader loader = null;
            if (saved_model_proto.MetaGraphs.Count == 1 && saved_model_proto.MetaGraphs[0].ObjectGraphDef is not null)
            {
                // skip python code: `metrics.IncrementReadApi(_LOAD_V2_LABEL)`
                var meta_graph_def = saved_model_proto.MetaGraphs[0];
                if (!BitConverter.IsLittleEndian)
                {
                    SavedModelUtils.swap_function_tensor_content(meta_graph_def);
                }

                var object_graph_proto = meta_graph_def.ObjectGraphDef;
                var ckpt_options = new CheckpointOptions(options.experimental_io_device);
                tf_with(ops.init_scope(), x =>
                {
                    loader = new Loader(object_graph_proto, saved_model_proto, export_dir, ckpt_options, options, filters);
                    root = (Trackable)loader.get(0);
                    // skip the assignment of `graph_debug_info`.
                });
                // skip the assignment of `tensorflow_version`
                // skip the assignment of `tensorflow_git_version`
                // skip the process of `metrics`.
            }
            else
            {
                if(filters is not null && filters.Count > 0)
                {
                    throw new ValueError("SavedModels saved from Tensorflow 1.x or Estimator (any"
                       + " version) cannot be loaded with node filters.");
                }
                tf_with(ops.init_scope(), x =>
                {
                    throw new NotImplementedException("Not implemented, please submit an issue to https://github.com/SciSharp/TensorFlow.NET/issues.");
                });
            }
            if(filters != null && filters.Count > 0)
            {
                return filters.Keys.ToDictionary(x => x, x => (Trackable)loader.get(x));
            }
            else
            {
                var res = new Dictionary<string, Trackable>();
                res["root"] = root;
                return res;
            }
        }

        public static (SavedModel, object?) parse_saved_model_with_debug_info(string export_dir)
        {
            var saved_model = parse_saved_model(export_dir);

            // TODO: implement debug info.

            return (saved_model, null);
        }

    }
}
