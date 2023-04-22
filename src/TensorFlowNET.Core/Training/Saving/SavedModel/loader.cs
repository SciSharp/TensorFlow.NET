using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Net.Sockets;
using System.Text;
using Tensorflow.Checkpoint;
using Tensorflow.Train;
using Tensorflow.Training;
using pbc = global::Google.Protobuf.Collections;
using static Tensorflow.Binding;
using System.Runtime.CompilerServices;
using Tensorflow.Variables;
using Tensorflow.Functions;
using Tensorflow.Training.Saving.SavedModel;
using Tensorflow.Trackables;
using OneOf;
using Tensorflow.Keras.Engine;

namespace Tensorflow
{
    /// <summary>
    /// Helper class to load an object-based SavedModel.
    /// </summary>
    public partial class Loader
    {
        private pbc::RepeatedField<global::Tensorflow.AssetFileDef> _asset_file_def;
        private Dictionary<string, pbc::MapField<string, AttrValue>> _operation_attributes;
        private SavedObjectGraph _proto;
        private string _export_dir;
        private CheckpointOptions _checkpoint_options;
        private LoadOptions _save_options;
        private IDictionary<string, (Trackable, Action<object, object, object>)> _node_filters;
        private Dictionary<string, int>? _node_path_to_id;
        private List<int>? _filtered_nodes;
        private List<int> _ordered_node_ids;
        private Dictionary<int, (Trackable, Action<object, object, object>)> _loaded_nodes;
        private List<object> _nodes;
        private Dictionary<int, Action<object, object, object>> _node_setters;
        private Dictionary<string, ConcreteFunction> _concrete_functions;
        private HashSet<string> _restored_concrete_functions;
        public Loader(SavedObjectGraph object_graph_proto, SavedModel saved_model_proto, string export_dir, 
            CheckpointOptions ckpt_options, LoadOptions save_options, IDictionary<string, (Trackable, Action<object, object, object>)> filters)
        {
            var meta_graph = saved_model_proto.MetaGraphs[0];
            _asset_file_def = meta_graph.AssetFileDef;
            _operation_attributes = meta_graph.GraphDef.Node.ToDictionary(x => x.Name, x => x.Attr);
            _proto = object_graph_proto;
            _export_dir = export_dir;
            // TODO(Rinne): This method is a bit slow (especially under debug mode), may need to be accelareted.
            _concrete_functions = function_deserialization.load_function_def_library(
                meta_graph.GraphDef.Library, _proto);
            _restored_concrete_functions = new HashSet<string>();
            _checkpoint_options = ckpt_options;
            _save_options = save_options;

            // TODO: `this._pretty_printer`

            _node_filters = filters;
            _node_path_to_id = _convert_node_paths_to_ints();
            _loaded_nodes = new Dictionary<int, (Trackable, Action<object, object, object>)>();

            if (filters != null)
            {
                foreach (var filter in filters)
                {
                    _loaded_nodes[_node_path_to_id[filter.Key]] = filter.Value;
                }
            }

            _filtered_nodes = _retrieve_all_filtered_nodes();

            _ordered_node_ids = _generate_ordered_node_ids();

            _load_all();


            if (!save_options.experimental_skip_checkpoint)
            {
                _restore_checkpoint();
            }
            foreach(var node in _nodes)
            {
                // skip the process of `CapturableResource`.
            }
        }

        /// <summary>
        /// Maps all string node paths in node_filters to the int node ids.
        /// </summary>
        /// <returns></returns>
        private Dictionary<string, int>? _convert_node_paths_to_ints()
        {
            if( _node_filters is null)
            {
                return null;
            }
            Dictionary<string, int> path_to_int = new();
            foreach(var node_id in _node_filters.Keys)
            {
                int int_node_id;
                var node_path = node_id.Split('.');
                if (node_path[0] != "root")
                {
                    throw new ValueError($"When passing string identifiers to node_filters, the first name" +
                        $" must be root. Received {node_path[0]}.");
                }
                int_node_id = 0;
                for(int i = 0; i < node_path.Length - 1; i++)
                {
                    var name = node_path[i + 1];
                    int_node_id = _find_node_child(int_node_id, name, String.Join(".", node_path.Take(i + 1)));
                }
                path_to_int[node_id] = int_node_id;
            }
            return path_to_int;
        }

        private int _find_node_child(int node_id, string child_name, string path)
        {
            foreach(var refer in _proto.Nodes[node_id].Children)
            {
                if(refer.LocalName == child_name)
                {
                    return refer.NodeId;
                }
            }
            throw new ValueError($"Unable to find node {path}.");
        }

        private List<int>? _retrieve_all_filtered_nodes()
        {
            if(_node_filters is null)
            {
                return null;
            }

            HashSet<int> all_filtered_nodes = new();
            Queue<string> nodes_to_visit = new Queue<string>(_node_filters.Keys);

            while(nodes_to_visit.Count > 0)
            {
                var node_path = nodes_to_visit.Dequeue();
                var node_id = _node_path_to_id[node_path];
                if (all_filtered_nodes.Contains(node_id))
                {
                    continue;
                }
                all_filtered_nodes.Add(node_id);
                Trackable node = null;
                Action<object, object, object> setter = null;
                if(_loaded_nodes.TryGetValue(node_id, out var res))
                {
                    (node, setter) = res;
                }
                if(node is not null)
                {
                    node._maybe_initialize_trackable();
                }

                foreach(var refer in _proto.Nodes[node_id].Children)
                {
                    Trackable children_object = null;
                    if(_loaded_nodes.TryGetValue(refer.NodeId, out var result))
                    {
                        children_object = result.Item1;
                    }
                    // See if node already tracks the child reference, in which case add the child to the loaded_nodes dict.
                    if(children_object is null && node is not null)
                    {
                        children_object = node._lookup_dependency(refer.LocalName);
                        if(children_object is TrackableDataStructure)
                        {
                            // TODO: set setter as lambda.

                            _loaded_nodes[refer.NodeId] = (children_object, setter);
                        }
                    }
                    string child_path = $"{node_path}.{refer.LocalName}";
                    _node_path_to_id[child_path] = refer.NodeId;
                    nodes_to_visit.Enqueue(child_path);
                }
            }

            if (all_filtered_nodes.Contains(0))
            {
                return null;
            }
            return all_filtered_nodes.ToList();
        }

        /// <summary>
        /// Orders the node ids so that dependencies appear first.
        /// </summary>
        /// <returns></returns>
        private List<int> _generate_ordered_node_ids()
        {
            List<int> unordered_ids;
            if(_filtered_nodes is null)
            {
                unordered_ids = Enumerable.Range(0, _proto.Nodes.Count).ToList();
            }
            else
            {
                unordered_ids = new List<int>(_filtered_nodes);
            }

            Dictionary<int, List<int>> dependency_map = new();
            foreach(var node_id in unordered_ids)
            {
                var deps = dependency_map.SetDefault(node_id, new List<int>());
                if (_loaded_nodes.ContainsKey(node_id))
                {
                    continue;
                }
                var proto = _proto.Nodes[node_id];
                foreach (var dep in _get_node_dependencies(proto).Values.Distinct())
                {
                    deps.Add(dep);
                    if(_filtered_nodes is not null && !_filtered_nodes.Contains(dep))
                    {
                        // TODO: add info with `_pretty_printer`.
                        throw new ValueError($"Unable to partially load SavedModel since the specified filter " +
                            $"does not include all required objects for loading (e.g. " +
                            $"variables used in functions or deserialization dependencies). " +
                            $"Please include this path in the filter: {dep}");
                    }
                }
                int? prev_slot = null;
                foreach(var slot_variable_proto in proto.SlotVariables)
                {
                    var slot_variable_node_id = slot_variable_proto.SlotVariableNodeId;
                    // The optimizer and original variable must be created before the slot
                    // variable, since the slot variable is generated using the Optimizer's
                    // add_slot API.
                    var slot_deps = dependency_map.SetDefault(slot_variable_node_id, new List<int>());
                    slot_deps.Add(node_id);
                    slot_deps.Add(slot_variable_proto.OriginalVariableNodeId);

                    if(prev_slot is not null)
                    {
                        slot_deps.Add(prev_slot.Value);
                    }
                    prev_slot = slot_variable_node_id;
                }
            }
            try
            {
                int total = 0;
                foreach(var v in dependency_map.Values)
                {
                    total += v.Count;
                }
                return TrackableUtils.order_by_dependency(dependency_map);
            }
            catch (TrackableUtils.CyclicDependencyError ex)
            {
                throw new ValueError("Encountered a cycle in the deserialization dependencies" +
                    "in the SavedModel. This is extremely unexpected, please" +
                    "file a bug and make sure you are not manually modifying the SavedModel.");
            }
        }

        /// <summary>
        /// Returns a dictionary of all dependencies of an object.
        /// </summary>
        /// <param name="proto"></param>
        /// <returns></returns>
        private Dictionary<OneOf<string, int>, int> _get_node_dependencies(SavedObject proto)
        {
            Dictionary<OneOf<string, int>, int> dependencies = new();
            foreach(var refer in proto.Dependencies)
            {
                dependencies[refer.LocalName] = refer.NodeId;
            }
            if(proto.KindCase == SavedObject.KindOneofCase.Function)
            {
                var concreete_functions = proto.Function.ConcreteFunctions;
                foreach(var fn_name in concreete_functions)
                {
                    foreach(var bound_input in _proto.ConcreteFunctions[fn_name].BoundInputs)
                    {
                        dependencies[bound_input] = bound_input;
                    }
                }
            }
            else if(proto.KindCase == SavedObject.KindOneofCase.BareConcreteFunction)
            {
                var fn_name = proto.BareConcreteFunction.ConcreteFunctionName;
                foreach(var bound_input in _proto.ConcreteFunctions[fn_name].BoundInputs)
                {
                    dependencies[bound_input] = bound_input;
                }
            }
            else if(proto.KindCase == SavedObject.KindOneofCase.Resource)
            {
                foreach(var child in proto.Children)
                {
                    if(child.LocalName == "_create_resource")
                    {
                        dependencies["_create_resource"] = child.NodeId;
                    }
                }
            }
            return dependencies;
        }

        /// <summary>
        /// Loads all nodes and functions from the SavedModel and their edges.
        /// </summary>
        private void _load_all()
        {
            _load_nodes();
            _load_edges();

            _setup_remaining_functions();
            _load_checkpoint_save_and_restore_functions();
        }

        /// <summary>
        /// Restores the checkpoint-related save/restore functions to all nodes.
        /// </summary>
        private void _load_checkpoint_save_and_restore_functions()
        {
            foreach(var (node_id, proto) in _iter_all_nodes())
            {
                var node = get(node_id);
                if(proto.SaveableObjects.Keys.Count == 1 && proto.SaveableObjects.First().Key == TrackableUtils.SERIALIZE_TO_TENSORS_NAME)
                {
                    // Restore Trackable serialize- and restore-from-tensor functions.
                    Debug.Assert(proto.SaveableObjects.Count == 1);
                    var saveable_object_proto = proto.SaveableObjects.Values.First();
                    var save_fn_id = saveable_object_proto.SaveFunction;
                    var restore_fn_id = saveable_object_proto.RestoreFunction;

                    throw new NotImplementedException("Not implemented, please submit an issue to https://github.com/SciSharp/TensorFlow.NET/issues");
                }
                else
                {
                    // Restore legacy SaveableObject functions.
                    Dictionary<string, (Trackable, Trackable)> saveable_fn_by_name = new();
                    foreach(var item in proto.SaveableObjects)
                    {
                        var name = item.Key;
                        var saveable_object_proto = item.Value;
                        var save_fn_id = saveable_object_proto.SaveFunction;
                        var restore_fn_id = saveable_object_proto.RestoreFunction;
                        saveable_fn_by_name[name] = ((Trackable)get(save_fn_id), (Trackable)get(restore_fn_id));
                    }
                    var saveable_objects = saveable_object_util.recreate_saveable_objects(saveable_fn_by_name, null);
                    if (saveable_objects is not null && saveable_objects.Count > 0)
                    {
                        if(node is Trackable trackable)
                        {
                            trackable.SelfSaveableObjectFactories = saveable_objects;
                        }
                        else
                        {
                            throw new TypeError();
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Load all saved objects.
        /// </summary>
        private void _load_nodes()
        {
            // `nodes` maps from node ids to recreated objects
            // `node_setters` maps from node ids to setter functions
            // (same signature as setattr) for setting children.
            var (nodes, node_setters) = _initialize_loaded_nodes();

            Dictionary<int, (int, global::Tensorflow.TrackableObjectGraph.Types.TrackableObject.Types.SlotVariableReference)>
                slot_variable_node_ids = new();

            foreach(var (node_id, proto) in _iter_all_nodes())
            {
                foreach(var slot_variable_proto in proto.SlotVariables)
                {
                    var slot_variable_node_id = slot_variable_proto.SlotVariableNodeId;
                    slot_variable_node_ids[slot_variable_node_id] = (node_id, slot_variable_proto);
                }
            }

            // Re-create everything.
            foreach (var (node_id, proto) in _iter_all_nodes())
            {
                if (nodes.ContainsKey(node_id))
                {
                    continue;
                }
                else if (slot_variable_node_ids.ContainsKey(node_id))
                {
                    // Use the public Optimizer interface when creating slot variables.
                    var (optimizer_node_id, slot_variable_proto) = slot_variable_node_ids[node_id];
                    var optimizer_object = nodes[optimizer_node_id] as IOptimizer;
                    var optimizer_variable = nodes[slot_variable_proto.OriginalVariableNodeId];

                    var slot_variable = optimizer_object.add_slot(optimizer_variable as IVariableV1, slot_variable_proto.SlotName);
                    nodes[slot_variable_proto.SlotVariableNodeId] = slot_variable as Trackable;
                    node_setters[slot_variable_proto.SlotVariableNodeId] = setattr;
                }
                else
                {
                    var (node, setter) = _recreate(proto, node_id, nodes);
                    nodes[node_id] = node;
                    node_setters[node_id] = setter;
                }
            }

            if (!nodes.ContainsKey(0))
            {
                nodes[0] = _recreate_base_user_object().Item1;
            }
            _nodes = new List<object>();
            for(int i = 0; i < _proto.Nodes.Count; i++)
            {
                _nodes.Add(nodes[i]);
            }
            _node_setters = node_setters;
        }

        /// <summary>
        /// Load state from checkpoint into the deserialized objects.
        /// </summary>
        private void _restore_checkpoint()
        {
            var variables_path = SavedModelUtils.get_variables_path(_export_dir);
            var saver = new TrackableSaver(new ObjectGraphView((Trackable)get(0)));
            tf_with(ops.device("CPU"), _ =>
            {
                saver.FilePrefixPlaceHolder = constant_op.constant(variables_path);
            });
            LoadStatus load_status;
            if (_save_options.allow_partial_checkpoint)
            {
                load_status = saver.restore(variables_path, _checkpoint_options).expect_partial();
                load_status.assert_nontrivial_match();
            }
            else
            {
                load_status = saver.restore(variables_path, _checkpoint_options);
                load_status.assert_existing_objects_matched();
            }
            var ckpt = (load_status as CheckpointLoadStatus).Checkpoint;

            if (!tf.Context.executing_eagerly())
            {
                throw new NotImplementedException("The checkpoint restore has not supported graph mode. " +
                    "Please submit an issue to https://github.com/SciSharp/TensorFlow.NET/issues");
            }
        }

        /// <summary>
        /// Adds edges from objects to other objects and functions.
        /// </summary>
        private void _load_edges()
        {
            foreach(var (node_id, object_proto) in _iter_all_nodes())
            {
                _add_object_graph_edges(object_proto, node_id);
            }

            if(_filtered_nodes is not null && _filtered_nodes.Contains(0))
            {
                var root = get(0);
                foreach(var node_path in _node_filters.Keys)
                {
                    var loaded_node = _nodes[_node_path_to_id[node_path]];

                    var path = node_path.Split('.');
                    var current_node = root;
                    foreach(var name in path.Skip(1).Take(path.Length - 2))
                    {
                        // `hasattr` and `setattr` is used here
                        throw new NotImplementedException();
                    }
                    // `hasattr` and `setattr` is used here
                    throw new NotImplementedException();
                }
            }
        }

        private void _setup_function_captures(string concrete_function_name, IDictionary<OneOf<string, int>, object> nodes)
        {
            if (_restored_concrete_functions.Contains(concrete_function_name))
            {
                return;
            }
            _restored_concrete_functions.Add(concrete_function_name);
            var concrete_function = _concrete_functions[concrete_function_name];
            var proto = _proto.ConcreteFunctions[concrete_function_name];
            var inputs = proto.BoundInputs.Select(x => nodes[x]);
            function_saved_model_utils.restore_captures(concrete_function, inputs);
        }

        private void _setup_remaining_functions()
        {
           // TODO: implement it with concrete functions.
        }

        public object get(int node_id)
        {
            return _nodes[node_id];
        }

        public object get(string node_id)
        {
            return get(_node_path_to_id[node_id]);
        }

        /// <summary>
        /// Adds edges from an object to its children.
        /// </summary>
        /// <param name="proto"></param>
        /// <param name="node_id"></param>
        private void _add_object_graph_edges(SavedObject proto, int node_id)
        {
            var obj = _nodes[node_id];
            var setter = _node_setters[node_id];

            foreach(var refer in proto.Children)
            {
                setter.Invoke(obj, refer.LocalName, _nodes[refer.NodeId]);
                // TODO(Rinne): deal with "__call__"
            }
        }

        private (Dictionary<int, object>, Dictionary<int, Action<object, object, object>>) _initialize_loaded_nodes()
        {
            Dictionary<int, object> nodes = new();
            Dictionary<int, Action<object, object, object>> node_setters = new();
            foreach(var item in _loaded_nodes)
            {
                var node_id = item.Key;
                var (node, setter) = item.Value;
                nodes[node_id] = node;
                node_setters[node_id] = setter;
            }
            return (nodes, node_setters);
        }

        private IEnumerable<(int, SavedObject)> _iter_all_nodes()
        {
            foreach(var node_id in _ordered_node_ids)
            {
                yield return (node_id, _proto.Nodes[node_id]);
            }
        }

        private (object, Action<object, object, object>) _recreate(SavedObject proto, int node_id, IDictionary<int, object> nodes)
        {
            // skip the registered classes.
            Dictionary<OneOf<string, int>, object> dependencies = new();
            foreach(var item in _get_node_dependencies(proto))
            {
                dependencies[item.Key] = nodes[item.Value];
            }

            return proto.KindCase switch
            {
                SavedObject.KindOneofCase.Resource => RestoredResource.deserialize_from_proto(proto, _operation_attributes),
                SavedObject.KindOneofCase.Asset => AssetResource.deserialize_from_proto(proto, _export_dir, _asset_file_def, _operation_attributes),
                SavedObject.KindOneofCase.Constant => TrackableConstant.deserialize_from_proto(proto, _operation_attributes),
                _ => _recreate_default(proto, node_id, dependencies)
            };
        }

        /// <summary>
        /// Creates a Python object from a SavedObject protocol buffer.
        /// </summary>
        /// <param name="proto"></param>
        /// <param name="node_id"></param>
        /// <param name="dependencies"></param>
        private (Trackable, Action<object, object, object>) _recreate_default(SavedObject proto, int node_id, IDictionary<OneOf<string, int>, object> dependencies)
        {
            return proto.KindCase switch
            {
                SavedObject.KindOneofCase.UserObject => _recreate_user_object(proto.UserObject, node_id),
                SavedObject.KindOneofCase.Function => _recreate_function(proto.Function, dependencies),
                SavedObject.KindOneofCase.BareConcreteFunction => _recreate_bare_concrete_function(proto.BareConcreteFunction, dependencies),
                SavedObject.KindOneofCase.Variable => _recreate_variable(proto.Variable),
                SavedObject.KindOneofCase.CapturedTensor => throw new NotImplementedException(),
                _ => throw new NotImplementedException()
            };
        }

        private (Trackable, Action<object, object, object>) _recreate_user_object(SavedUserObject? proto, int node_id)
        {
            // skip the check of proto identifier because of lack of property.
            var (trackable, setter) = RevivedTypes.deserialize(proto);
            if(trackable is null)
            {
                return _recreate_base_user_object(proto, node_id);
            }
            return (trackable, setter);
        }

        private (Trackable, Action<object, object, object>) _recreate_base_user_object(SavedUserObject? proto = null, int? node_id = null)
        {
            return (new _UserObject(), setattr);
        }

        private (BaseResourceVariable, Action<object, object, object>) _recreate_variable(SavedVariable proto)
        {
            string name = proto.Name;
            string dbg_name = !string.IsNullOrEmpty(name) ? name : "<variable loaded from saved model>";

            // TODO(Rinne): `validate_synchronization_aggregation_trainable`

            var (synchronization, aggregation, trainable) = ResourceVariable.validate_synchronization_aggregation_trainable(
                proto.Synchronization, proto.Aggregation, proto.Trainable, dbg_name);

            var saved_device = proto.Device;
            var load_with_device = _save_options.experimental_variable_policy.save_variable_devices() && !string.IsNullOrEmpty(saved_device);

            if (load_with_device)
            {
                return tf_with(ops.device(saved_device), _ =>
                {
                    return (new UninitializedVariable(
                        shape: new Shape(proto.Shape.Dim.Select(x => (int)x.Size).ToArray()),
                        dtype: (TF_DataType)proto.Dtype,
                        name: name,
                        trainable: trainable,
                        aggregation: aggregation
                    ), setattr);
                });
            }
            else
            {
                return (new UninitializedVariable(
                    shape: new Shape(proto.Shape.Dim.Select(x => (int)x.Size).ToArray()),
                    dtype: (TF_DataType)proto.Dtype,
                    name: name,
                    trainable: trainable,
                    aggregation: aggregation
                ), setattr);
            }
        }

        private (Function, Action<object, object, object>) _recreate_function(SavedFunction proto,
            IDictionary<OneOf<string, int>, object> dependencies)
        {
            var fn = function_deserialization.recreate_function(proto, _concrete_functions);
            foreach (var name in proto.ConcreteFunctions)
            {
                _setup_function_captures(name, dependencies);
            }
            return (fn, setattr);
        }

        private (ConcreteFunction, Action<object, object, object>) _recreate_bare_concrete_function(SavedBareConcreteFunction proto,
            IDictionary<OneOf<string, int>, object> dependencies)
        {
            var fn = function_deserialization.setup_bare_concrete_function(proto, _concrete_functions);
            _setup_function_captures(proto.ConcreteFunctionName, dependencies);
            return (fn, setattr);
        }

        private (Tensor, Action<object, object, object>) _get_tensor_from_fn(CapturedTensor proto)
        {
            var outer_graph = _concrete_functions[proto.ConcreteFunction].func_graph;
            var captured_tensor = outer_graph.get_tensor_by_name(proto.Name);
            return (captured_tensor, setattr);
        }

        // TODO: remove this to a common class.
        public static Action<object, object, object> setattr = (x, y, z) =>
        {
            Debug.Assert(y is string);
            if(x is Trackable trackable)
            {
                trackable.SetAttr(y as string, z);
            }
            else
            {
                var properties = x.GetType().GetProperties();
                foreach (var p in properties)
                {
                    if ((string)y == p.Name)
                    {
                        p.SetValue(x, z);
                        return;
                    }
                }
            }
            // TODO(Rinne): check if the property has been set successfully.
            //throw new ValueError($"Cannot find the property {y} of {x}.");
        };

        public class _UserObject: AutoTrackable
        {

        }
    }
}
