using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Tensorflow.Checkpoint;
using Tensorflow.Contexts;
using Tensorflow.Functions;
using Tensorflow.Train;
using Tensorflow.Training;
using pbc = global::Google.Protobuf.Collections;
using static Tensorflow.Binding;
using Tensorflow.Training.Saving.SavedModel;

namespace Tensorflow;

public class SaveableView
{
    private AugmentedGraphView _augmented_graph_view;
    private SaveOptions _options;
    private IList<Trackable> _trackable_objects;
    private List<Trackable> _nodes;
    private IDictionary<Trackable, IEnumerable<TrackableReference>> _node_paths;
    private IDictionary<Trackable, int> _node_ids;
    private IDictionary<Trackable, pbc::RepeatedField<global::Tensorflow.TrackableObjectGraph.Types.TrackableObject.Types.SlotVariableReference>>
        _slot_variables;
    private IDictionary<Trackable, string> _object_names;
    private List<object> _gradient_functions; // to be completed
    private List<RegisteredGradient> _gradient_defs; // to be completed
    private List<ConcreteFunction> _concrete_functions;
    private Dictionary<Tensor, int> _captured_tensor_node_ids;
    private Dictionary<Trackable, IDictionary<string, ConcreteFunction>> _saveable_objects_map;
    private Dictionary<Trackable, string> _obj_to_registered_saver;

    public AugmentedGraphView AugmentedGraphView
    {
        get => _augmented_graph_view;
    }
    
    public Trackable Root
    {
        get => _nodes[0];
    }
    public List<Trackable> Nodes
    {
        get => _nodes;
    }
    public IDictionary<Trackable, int> NodeIds
    {
        get => _node_ids;
    }
    public List<RegisteredGradient> GradientDefs
    {
        get => _gradient_defs;
    }
    public IDictionary<Trackable, IEnumerable<TrackableReference>> NodePaths
    {
        get => _node_paths;
    }
    public SaveableView(AugmentedGraphView augmented_graph_view, SaveOptions options)
    {
        _augmented_graph_view = augmented_graph_view;
        _options = options;

        (_trackable_objects, _node_paths, _node_ids, _slot_variables, _object_names) =
            CheckPointUtils.objects_ids_and_slot_variables_and_paths(_augmented_graph_view);
        
        // TODO: deal with untraced functions.
        
        initialize_save_and_restore_functions();
        initialize_nodes_and_concrete_functions();

        _captured_tensor_node_ids = new();
    }

    private void initialize_save_and_restore_functions()
    {
        // TODO: deal with the return value of `get_checkpoint_factories_and_keys`.
        var (checkpoint_factory_map, registered_savers) = SaveUtilV1.get_checkpoint_factories_and_keys(_object_names);
        // skip the process of registered savers and the generation of saveable_objects_map and _obj_to_registered_saver.
        _obj_to_registered_saver = new();
        _saveable_objects_map = new();
    }

    private void initialize_nodes_and_concrete_functions()
    {
        _nodes = _trackable_objects.ToList().ConvertAll(x => x); // deep copy
        _gradient_functions = new();
        _gradient_defs = new();

        // TODO: deal with the condition that obj in `_saveable_objects_map`.
        // foreach (var obj in _nodes)
        // {
        //     
        // }

        //_concrete_functions = new();
        //foreach (var obj in _nodes)
        //{
        //    if (obj is ConcreteFunction)
        //    {
        //        _concrete_functions.Add((ConcreteFunction)obj);
        //    }
        //}
    }

    public List<ConcreteFunction> get_concrete_resource_initializers()
    {
        // TODO: complete the implementation.
        return new List<ConcreteFunction>();
    }
    
    public (Dictionary<Trackable, Trackable>, Dictionary<Tensor, Tensor>, AssetInfo) map_resources()
    {
        Debug.Assert(!tf.Context.executing_eagerly());

        Dictionary<Trackable, Trackable> object_map = new();
        Dictionary<Tensor, Tensor> tensor_map = new();

        AssetInfo assetInfo = new(new List<AssetFileDef>(), new Dictionary<object, object>(),
            new Dictionary<AssetInfo, string>(), new Dictionary<object, object>());

        foreach (var node_id in dependency_sorted_node_ids())
        {
            var obj = _nodes[node_id];
            var tensors = obj.export_to_saved_model_graph(object_map, tensor_map, _options);
            // TODO: deal with Asset (if obj is Asset)
            foreach (var tensor in tensors)
            {
                _captured_tensor_node_ids[tensor] = node_id;
            }
        }

        return (object_map, tensor_map, assetInfo);
    }

    /// <summary>
    /// Returns topologically sorted nodes, sorted by dependencies.
    /// </summary>
    public List<int> dependency_sorted_node_ids()
    {
        Dictionary<int, List<int>> dependency_map = new();
        foreach (var node in _nodes)
        {
            var node_id = _node_ids[node];
            List<int> deps = new List<int>();
            dependency_map.Add(node_id, deps);
            
            // TODO: deal with captured tensor.

            foreach (var (_, dep) in _augmented_graph_view.list_dependencies(node))
            {
                if (!_node_ids.ContainsKey(dep))
                {
                    var node_path = TrackableUtils.pretty_print_node_path(_node_paths[node]);
                    throw new ValueError(
                        $"Found an untracked dependency. Object {node_path} depends on {dep}, " +
                        $"but this dependency isn't listed as a child. Please track this child by " +
                        $"overriding `_trackable_children` or use `._track_trackable`.");
                }
                deps.Add(_node_ids[dep]);
            }
        }

        try
        {
            return TrackableUtils.order_by_dependency(dependency_map);
        }
        catch (TrackableUtils.CyclicDependencyError err)
        {
            List<string> pretty_printed_nodes = new();
            List<string> pretty_printed_dependencies = new();

            foreach (var pair in err.LeftOverDependencyMap)
            {
                var x = pair.Key;
                var deps = pair.Value;
                var node_path = TrackableUtils.pretty_print_node_path(_node_paths[_nodes[x]]);
                pretty_printed_nodes.Add($"\tNode {x.ToString()} = {node_path} (type {_nodes[x]})");
                pretty_printed_dependencies.Add(
                    $"\tNode {x.ToString()} depends on nodes [{string.Join(", ", deps.Select(x => x.ToString()))}]");
            }

            throw new ValueError($"There is one or more dependency cycle in the saved Trackable object. " +
                                 $"Saving cannot continue until this cycle is resolved." +
                                 $"\n>> Unresolved nodes:\n{string.Join("\n", pretty_printed_nodes)}" +
                                 $"\n>> Unresolved cyclic dependencies:\n{string.Join("\n", pretty_printed_dependencies)}");
        }
    }

    /// <summary>
    /// Corresponding to tensorflow/python/saved_model/save.py/_serialize_object_graph
    /// </summary>
    /// <param name="asset_index"></param>
    /// <returns></returns>
    public SavedObjectGraph serialize_object_graph(IDictionary<object, object> asset_file_def_index)
    {
        SavedObjectGraph proto = new();
        fill_object_graph_proto(proto);
        
        // TODO: complete the process of concrete functions.

        int cnt = Math.Min(_nodes.Count, proto.Nodes.Count);
        for (int i = 0; i < cnt; i++)
        {
            var obj = _nodes[i];
            var obj_proto = proto.Nodes[i];
            write_object_proto(obj, obj_proto, asset_file_def_index, x => _augmented_graph_view.list_children(x));
        }

        return proto;
    }

    private static void write_object_proto(Trackable obj, SavedObject proto,
        IDictionary<object, object> asset_file_def_index, Func<Trackable, List<TrackableReference>> list_children_fn)
    {
        // skip the process of type Asset
        if (resource_variable_ops.is_resource_variable(obj))
        {
            var options = SaveContext.get_save_options();
            (obj as BaseResourceVariable).write_object_proto(proto, options);
        }
        else if (obj is Function)
        {
            // TODO: complete it.
            throw new NotImplementedException();
        }
        else if (obj is ConcreteFunction)
        {
            // TODO(Rinne): complete it.
            // throw new NotImplementedException();
        }
        // skip the process of type `_CapturedTensor` and `CapturableResource`.
        else
        {
            var registered_type_proto = RevivedTypes.serialize(obj);
            if (registered_type_proto is null)
            {
                registered_type_proto = new SavedUserObject()
                {
                    Identifier = obj.ObjectIdentifier,
                    Version = new VersionDef()
                    {
                        Producer = 1,
                        MinConsumer = 1,
                        BadConsumers = { }
                    }
                };
            }

            proto.UserObject = new SavedUserObject(registered_type_proto);
        }
        
        // TODO: try get the registered_name from `registration`.
    }

    public void fill_object_graph_proto(SavedObjectGraph proto)
    {
        for (int node_id = 0; node_id < _nodes.Count; node_id++)
        {
            var node = _nodes[node_id];
            Debug.Assert(_node_ids[node] == node_id);
            SavedObject object_proto = new();
            if (_slot_variables.TryGetValue(node, out var value))
            {
                object_proto.SlotVariables.AddRange(value);
            }
            // skip the check of type `_CapturedTensor`
            foreach (var child in _augmented_graph_view.list_children(node))
            {
                var child_proto = new TrackableObjectGraph.Types.TrackableObject.Types.ObjectReference();
                child_proto.NodeId = _node_ids[child.Refer];
                child_proto.LocalName = child.Name;
                object_proto.Children.Add(child_proto);
            }

            foreach (var pair in _augmented_graph_view.list_dependencies(node))
            {
                var child_proto = new TrackableObjectGraph.Types.TrackableObject.Types.ObjectReference();
                child_proto.NodeId = _node_ids[pair.Item2];
                child_proto.LocalName = pair.Item1;
                object_proto.Dependencies.Add(child_proto);
            }

            if (_saveable_objects_map.ContainsKey(node))
            {
                // TODO: complete it.
                throw new NotImplementedException();
            }
            else if(_obj_to_registered_saver.ContainsKey(node))
            {
                // TODO: complete it.
                // We now skip it for the lack of `SavedObject.registered_saver` API.
                throw new NotImplementedException();
            }

            proto.Nodes.Add(object_proto);
        }
    }
}
