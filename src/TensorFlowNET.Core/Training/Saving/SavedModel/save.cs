using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Google.Protobuf;
using Tensorflow.Checkpoint;
using Tensorflow.Functions;
using Tensorflow.Train;
using Tensorflow.Exceptions;
using static Tensorflow.Binding;
using Tensorflow.Training.Saving.SavedModel;

namespace Tensorflow;

public static partial class SavedModelUtils
{
    private static readonly IEnumerable<int> byte_swappable = new List<TF_DataType>()
    {
        dtypes.float16, dtypes.float32, dtypes.float64, TF_DataType.TF_BFLOAT16,
        dtypes.complex64, dtypes.complex128, TF_DataType.TF_UINT16, dtypes.uint32,
        dtypes.uint64, TF_DataType.TF_INT16, dtypes.int32, dtypes.int64, TF_DataType.TF_QINT16,
        TF_DataType.TF_QUINT16, TF_DataType.TF_QINT32
    }.Select(x => (int)x);
    
    public static (IList<Trackable>, IDictionary<Trackable, IEnumerable<TrackableReference>>) save_and_return_nodes(Trackable obj, 
        string export_dir, ConcreteFunction? signatures, SaveOptions? options = null, bool experimental_skip_checkpoint = false)
    {
        if (options is null)
        {
            options = new SaveOptions();
        }

        var saved_model = new Tensorflow.SavedModel();
        var meta_graph_def = new MetaGraphDef();
        saved_model.MetaGraphs.Add(meta_graph_def);

        var (_, exported_graph, object_saver, asset_info, saved_nodes, node_paths) =
            _build_meta_graph(obj, signatures, options, meta_graph_def);
        saved_model.SavedModelSchemaVersion = Tensorflow.Constants.SAVED_MODEL_SCHEMA_VERSION;

        if (!experimental_skip_checkpoint)
        {
            SavedModelUtils.get_or_create_variables_dir(export_dir);
            CheckpointOptions ckpt_options = new(options.experimental_io_device);
            object_saver.save(SavedModelUtils.get_variables_path(export_dir), options:ckpt_options);
        }
        BuilderUtils.copy_assets_to_destination_dir(asset_info.asset_filename_map, export_dir);

        if (tf.Context.executing_eagerly())
        {
            // tensorflow python has a check of `context.async_wait()` here.
        }
        
        // TODO: deal with `pywrap_saved_model.Save(export_dir)`.

        var saved_model_serialized = saved_model.ToString();

        // This is a state depending on some py-c APIs. Here we temporarily set it as `true`.
        if (true)
        {
            var fingerprint_path = Path.Combine(tf.compat.as_str(export_dir),
                tf.compat.as_str(Constants.FINGERPRINT_FILENAME));
            // TODO: add c api and complete the fingerprint def.
            var fingerprint_proto = "";
            File.WriteAllText(fingerprint_path, fingerprint_proto);
        }

        var path = Path.Combine(tf.compat.as_str(export_dir), tf.compat.as_str(Constants.SAVED_MODEL_FILENAME_PB));
        File.WriteAllBytes(path, saved_model.ToByteArray());
        //File.WriteAllText(path, saved_model.ToString());

        if (options.save_debug_info)
        {
            throw new NotImplementedException();
        }
        
        ops.dismantle_graph(exported_graph);

        return (saved_nodes, node_paths);
    }

    private static (MetaGraphDef, Graph, TrackableSaver, AssetInfo, IList<Trackable>,
        IDictionary<Trackable, IEnumerable<TrackableReference>>) _build_meta_graph(Trackable obj,
            ConcreteFunction? signatures, SaveOptions options, MetaGraphDef? meta_graph_def = null)
    {
        using (SaveContext.save_context(options))
        {
            if (ops.inside_function())
            {
                throw new AssertionError("`tf.saved_model.save` is not supported inside a traced [AutoGraph]. " +
                                         "Move the call to the outer eagerly-executed context.");
            }

            if (meta_graph_def is null)
            {
                meta_graph_def = new MetaGraphDef();
            }

            AugmentedGraphView augmented_graph_view = new AugmentedGraphView(obj);
            if (signatures is null)
            {
                signatures = SignatureSerializationUtils.find_function_to_export(augmented_graph_view);
            }

            // TODO: process of aignatures and wrapped_functions

            SaveableView saveable_view = new SaveableView(augmented_graph_view, options);
            TrackableSaver object_saver = new TrackableSaver(augmented_graph_view);
            var (asset_info, exported_graph) = _fill_meta_graph_def(meta_graph_def, saveable_view, signatures,
                options.namespace_white_list, options.experimental_custom_gradients);
            if (options.function_aliases is not null)
            {
                var function_aliases = meta_graph_def.MetaInfoDef.FunctionAliases;
                foreach (var pair in options.function_aliases)
                {
                    var alias = pair.Key;
                    var func = pair.Value;
                    // TODO: complete it.
                    throw new NotImplementedException();
                }
            }

            var object_graph_proto = saveable_view.serialize_object_graph(asset_info.asset_index);
            meta_graph_def.ObjectGraphDef = new SavedObjectGraph(object_graph_proto);

            return (meta_graph_def, exported_graph, object_saver, asset_info, saveable_view.Nodes, saveable_view.NodePaths);
        }
    }

    private static (AssetInfo, Graph) _fill_meta_graph_def(MetaGraphDef meta_graph_def, SaveableView saveable_view,
        ConcreteFunction signatures, IEnumerable<string> namespace_whitelist,
        bool save_custom_gradients)
    {
        var resource_initializers = saveable_view.get_concrete_resource_initializers();
        var exported_graph = new Graph();

        Dictionary<Trackable, Trackable> object_map;
        Dictionary<Tensor, Tensor> tensor_map;
        AssetInfo asset_info;
        var g = exported_graph.as_default();
        (object_map, tensor_map, asset_info) = saveable_view.map_resources();
        // TODO: deal with signatures.
        if (save_custom_gradients)
        {
            // TODO: trace gradient functions.
        }

        foreach (var resource_initializer_function in resource_initializers)
        {
            // List<Trackable> asset_dependencies = new();
            // TODO: deal with initializers
        }

        // using(ops.control_dependencies(...))
        var init_op = control_flow_ops.no_op();
        if (meta_graph_def.CollectionDef.ContainsKey(Tensorflow.Constants.MAIN_OP_KEY))
        {
            meta_graph_def.CollectionDef[Tensorflow.Constants.MAIN_OP_KEY].NodeList.Value.Append(init_op.name);
        }
        else
        {
            meta_graph_def.CollectionDef[Tensorflow.Constants.MAIN_OP_KEY] = new CollectionDef();
        }
        // Lack `CopyFrom` API
        // meta_graph_def.SignatureDef[Tensorflow.Constants.INIT_OP_SIGNATURE_KEY]

        g.Exit();

        foreach (var obj in object_map.Values)
        {
            obj._maybe_initialize_trackable();
        }

        // TODO: add the implementation of `call_with_mapped_functions`.
        var (named_saveable_objects, registered_savers) =
            SaveUtilV1.frozen_saveables_and_savers(saveable_view.AugmentedGraphView, object_map, exported_graph, false);
        var saver = MultiDeviceSaver.from_saveables(named_saveable_objects, registered_savers, false);

        var eg = exported_graph.as_default();
        var saver_def = saver.to_proto();
        meta_graph_def.SaverDef = saver_def;
        eg.Exit();


        saveable_view.dependency_sorted_node_ids();

        var graph_def = exported_graph.as_graph_def(true);
        graph_def.Library.RegisteredGradients.AddRange(saveable_view.GradientDefs);
        verify_ops(graph_def, namespace_whitelist);

        meta_graph_def.GraphDef = new GraphDef(graph_def);
        meta_graph_def.MetaInfoDef = new();
        meta_graph_def.MetaInfoDef.Tags.Add(TagConstants.SERVING);
        meta_graph_def.MetaInfoDef.TensorflowVersion = tf.VERSION;
        // TODO: add git version.
        meta_graph_def.MetaInfoDef.TensorflowGitVersion = "";
        meta_graph_def.MetaInfoDef.StrippedDefaultAttrs = true;
        meta_graph_def.MetaInfoDef.StrippedOpList = new();
        meta_graph_def.MetaInfoDef.StrippedOpList.MergeFrom(meta_graph.stripped_op_list_for_graph(meta_graph_def.GraphDef));
        meta_graph_def.AssetFileDef.AddRange(asset_info.asset_defs);
        
        // TODO: deal with signatures here.
        
        meta_graph.strip_graph_default_valued_attrs(meta_graph_def);

        if (!BitConverter.IsLittleEndian)
        {
            swap_function_tensor_content(meta_graph_def);
        }

        return (asset_info, exported_graph);
    }

    private static void verify_ops(GraphDef graph_def, IEnumerable<string>? namespace_whitelist)
    {
        return;
        // if (namespace_whitelist is null || !namespace_whitelist.Any())
        // {
        //     return;
        // }
        
        // skip the check for the lack of `meta_graph.ops_used_by_graph_def`.
    }

    public static void swap_function_tensor_content(MetaGraphDef meta_graph_def)
    {
        var functions = meta_graph_def.GraphDef.Library.Function;
        foreach (var function in functions)
        {
            var node_def = function.NodeDef;
            foreach (var node in node_def)
            {
                if (node.Op == "Const")
                {
                    var tensor = node.Attr["value"].Tensor;
                    byte_swap_tensor_content(tensor);
                }
            }
        }
    }

    public static void byte_swap_tensor_content(TensorProto tensor)
    {
        if (byte_swappable.Contains((int)tensor.Dtype))
        {
            var tshape = tensor.TensorShape.Dim;
            var tensor_bytes = tensor.TensorContent;
            if (tensor_bytes is not null && !tensor_bytes.IsEmpty)
            {
                long tensor_size = 1;
                foreach (var sz in tshape)
                {
                    tensor_size *= sz.Size;
                }

                var chunksize = tensor_bytes.Length / tensor_size;
                List<byte> reversed_bytes = new();
                for (int i = 0; i < tensor_bytes.Length; i += (int)chunksize)
                {
                    var current = tensor_bytes.Skip(i).Take((int)chunksize).Reverse();
                    reversed_bytes.AddRange(current);
                }
                tensor.TensorContent = ByteString.CopyFrom(reversed_bytes.ToArray());
            }
        }
    }
}
