using Google.Protobuf;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Tensorflow.Contexts;
using Tensorflow.Eager;
using Tensorflow.Train;
using Tensorflow.Exceptions;
using static Tensorflow.TrackableObjectGraph.Types.TrackableObject.Types;
using static Tensorflow.Binding;
using Tensorflow.Operations;
using Newtonsoft.Json;
using Tensorflow.Training;
using OneOf;

namespace Tensorflow.Checkpoint;

/// <summary>
/// Saves and restores a `Trackable` object and its dependencies.
/// </summary>
public class TrackableSaver
{
    private ObjectGraphView _graph_view;
    private Tensor _cached_save_operation;
    private TrackableObjectGraph _last_save_object_graph;
    private Tensor? _object_graph_feed_tensor = null;
    private Tensor? _file_prefix_feed_tensor = null;
    private Tensor? _file_prefix_placeholder = null;
    private Dictionary<Trackable, Trackable>? _object_map = null;
    private object? _cache = null;
    public Tensor? FilePrefixPlaceHolder
    {
        get
        {
            return _file_prefix_placeholder;
        }
        set
        {
            _file_prefix_placeholder = value;
        }
    }
    public TrackableSaver(ObjectGraphView graph_view)
    {
        _graph_view = graph_view;
        
        // TODO: cache when not executing eagerly.
        // including `_cache`, `_file_prefix_feed_tensor`, `_file_prefix_placeholder`
        // `_object_graph_feed_tensor`, `_object_map`, `_restore_op_cache`, `_saveables_cache`
        
    }

    private (IDictionary<Trackable, IDictionary<string, IDictionary<string, OneOf<Tensor, SaveSpec>>>>, IDictionary<Tensor, object>, IDictionary<string, IDictionary<string, Trackable>>, TrackableObjectGraph) 
        gather_serialized_tensors(Tensor? object_graph_tensor = null)
    {
        var (serialized_tensors, feed_additions, registered_savers, graph_proto) = SaveUtil.serialize_graph_view(_graph_view, _object_map, cache:_cache);

        // TODO: cache.

        if(object_graph_tensor is null)
        {
            tf_with(ops.device("/cpu:0"), _ =>
            {
                object_graph_tensor = constant_op.constant(graph_proto.ToByteArray());
            });
        }
        else
        {
            feed_additions[object_graph_tensor] = graph_proto.ToByteArray();
        }
        Debug.Assert(!serialized_tensors.ContainsKey(Trackable.None) || !serialized_tensors[Trackable.None].ContainsKey(Trackable.Constants.OBJECT_GRAPH_PROTO_KEY));
        if (!serialized_tensors.ContainsKey(Trackable.None))
        {
            serialized_tensors[Trackable.None] = new Dictionary<string, IDictionary<string, OneOf<Tensor, SaveSpec>>>();
        }
        serialized_tensors[Trackable.None][Trackable.Constants.OBJECT_GRAPH_PROTO_KEY] = new Dictionary<string, OneOf<Tensor, SaveSpec>>();
        serialized_tensors[Trackable.None][Trackable.Constants.OBJECT_GRAPH_PROTO_KEY].Add(saveable_object_util.NO_SLICE_SPEC_KEY, object_graph_tensor);
        return (serialized_tensors, feed_additions, registered_savers, graph_proto);
    }

    private (Tensor, IDictionary<Tensor, object>) save_cached_when_graph_building(Tensor file_prefix, Tensor object_graph_tensor, CheckpointOptions options)
    {
        var (serialized_tensors, feed_additions, registered_savers, graph_proto) = gather_serialized_tensors(object_graph_tensor);

        Func<(Tensor, IDictionary<Tensor, object>)> run_save = () =>
        {
            if (_last_save_object_graph != graph_proto || tf.Context.executing_eagerly() || ops.inside_function())
            {
                var saver = new MultiDeviceSaver(serialized_tensors, registered_savers);
                var save_op = saver.save(file_prefix, options);

                // tensorflow python: `with ops.device("/cpu:0"):`
                using (ops.control_dependencies(new object[] { save_op }))
                {
                    _cached_save_operation = array_ops.identity(file_prefix);
                }
                _last_save_object_graph = graph_proto;
            }
            return (_cached_save_operation, feed_additions);
        };

        if (options.experimental_enable_async_checkpoint)
        {
            throw new NotImplementedException();
        }

        return run_save();
    }

    private (Tensor, IDictionary<Tensor, object>) save_cached_when_graph_building(string file_prefix, Tensor object_graph_tensor, CheckpointOptions options)
    {
        var (serialized_tensors, feed_additions, registered_savers, graph_proto) = gather_serialized_tensors(object_graph_tensor);

        Func<(Tensor, IDictionary<Tensor, object>)> run_save = () =>
        {
            if (_last_save_object_graph != graph_proto || tf.Context.executing_eagerly() || ops.inside_function())
            {
                var saver = new MultiDeviceSaver(serialized_tensors, registered_savers);
                var save_op = saver.save(file_prefix, options);

                // tensorflow python: `with ops.device("/cpu:0"):`
                using (ops.control_dependencies(new object[] {save_op} ))
                {
                    _cached_save_operation = array_ops.identity(tf.constant(file_prefix));
                }
                _last_save_object_graph = graph_proto;
            }
            return (_cached_save_operation, feed_additions);
        };

        if (options.experimental_enable_async_checkpoint)
        {
            throw new NotImplementedException();
        }

        return run_save();
    }

    // TODO: parameter write_done_callback
    public Tensor save(string file_prefix, int? checkpoint_number = null, Session? session = null,
        CheckpointOptions? options = null)
    {
        if (options is null)
        {
            options = new CheckpointOptions();
        }

        Dictionary<Tensor, object> feed_dict = new();
        bool use_session = (!tf.Context.executing_eagerly() && !ops.inside_function());
        if (checkpoint_number is not null)
        {
            file_prefix = $"{file_prefix}-{checkpoint_number?.ToString()}";
        }

        Tensor file_prefix_tensor;
        Tensor object_graph_tensor;
        string file_prefix_to_save;
        if (use_session)
        {
            if (_object_graph_feed_tensor is null)
            {
                // In python there is `with ops.device("/cpu:0")`.
                _object_graph_feed_tensor = constant_op.constant("", TF_DataType.TF_STRING);
                _file_prefix_feed_tensor = constant_op.constant("", TF_DataType.TF_STRING);
            }

            object_graph_tensor = _object_graph_feed_tensor;
            file_prefix_tensor = _file_prefix_feed_tensor;
            feed_dict[file_prefix_tensor] = file_prefix;
            file_prefix_to_save = "";
        }
        else
        {
            // In python there is `with ops.device("/cpu:0")`.
            file_prefix_tensor = ops.convert_to_tensor(file_prefix, TF_DataType.TF_STRING);
            object_graph_tensor = null;
            file_prefix_to_save = file_prefix;
        }

        var (save_path, new_feed_additions) =
            save_cached_when_graph_building(file_prefix_to_save, object_graph_tensor, options);

        if (new_feed_additions is not null)
        {
            foreach (var pair in new_feed_additions)
            {
                feed_dict.Add(pair.Key, pair.Value);
            }
        }
        if(!use_session)
        {
            session = null;
        }
        else if (session is null)
        {
            session = new Session(); // In python it uses `get_session`.
        }

        if (session is not null)
        {
            var s = feed_dict.Select(x => new FeedItem(x.Key, x.Value)).ToArray();
            return session.run((Tensor)save_path, s);
        }
        else if (use_session)
        {
            throw new RuntimeError($"Unable to save checkpoint to \"{file_prefix}\" " +
                                   "in graph mode without a default session. Please use " +
                                   "`with tf.Session():` to create a session.");
        }
        else
        {
            return save_path;
        }
    }

    public LoadStatus restore(string? save_path, CheckpointOptions? options = null)
    {
        if (options is null)
        {
            options = new CheckpointOptions();
        }
        if(save_path is null)
        {
            return new InitializationOnlyStatus(_graph_view, ops.uid());
        }

        CheckpointReader reader = new CheckpointReader(save_path);
        bool graph_building = tf.Context.executing_eagerly();
        Dictionary<string, TF_DataType> dtype_map = null;
        if (!graph_building)
        {
            dtype_map = reader.VariableToDataTypeMap;
        }
        Tensor object_graph_string = reader.GetTensor(Trackable.Constants.OBJECT_GRAPH_PROTO_KEY, dtype: TF_DataType.TF_STRING);

        Dictionary<Tensor, string> file_prefix_feed_dict;
        Tensor file_prefix_tensor = null;
        if (graph_building)
        {
            if(_file_prefix_placeholder is null)
            {
                _file_prefix_placeholder = tf_with(ops.device("/cpu:0"), _ =>
                {
                    return constant_op.constant("model");
                });
            }
            file_prefix_tensor = _file_prefix_placeholder;
            file_prefix_feed_dict = new();
            file_prefix_feed_dict[_file_prefix_placeholder] = save_path;
        }
        else
        {
            file_prefix_tensor = tf_with(ops.device("/cpu:0"), _ =>
            {
                return constant_op.constant(save_path);
            });
            file_prefix_feed_dict = null;
        }
        TrackableObjectGraph object_graph_proto = new();
        if(object_graph_string.ndim > 0)
        {
            object_graph_proto.MergeFrom(object_graph_string.BufferToArray());
        }
        else
        {
            object_graph_proto.MergeFrom(object_graph_string.StringBytes()[0]);
        }
        CheckpointRestoreCoordinator checkpoint = new CheckpointRestoreCoordinator(
            object_graph_proto: object_graph_proto, 
            save_path: save_path, 
            save_path_tensor: file_prefix_tensor,
            reader: reader,
            restore_op_cache: null,
            graph_view: _graph_view,
            options: options,
            saveables_cache: null
        );

        new CheckpointPosition(checkpoint, 0).restore(_graph_view.Root);

        if(_graph_view.AttachedDependencies is not null)
        {
            foreach(var refer in _graph_view.AttachedDependencies)
            {
                if(refer.Name == "root")
                {
                    continue;
                }
                int? proto_id = null;
                // Find proto ID of attached dependency (if it is in the proto).
                foreach (var proto_refer in object_graph_proto.Nodes[0].Children)
                {
                    if(proto_refer.LocalName == refer.Name)
                    {
                        proto_id = proto_refer.NodeId;
                        break;
                    }
                }

                if (proto_id is null)
                {
                    continue;
                }

                // Object has already been restored. This can happen when there's an
                // indirect connection from the attached object to the root.
                if (checkpoint.ObjectByProtoId.ContainsKey(proto_id.Value))
                {
                    continue;
                }

                new CheckpointPosition(checkpoint, proto_id.Value).restore(refer.Refer);
            }
        }

        return new CheckpointLoadStatus(checkpoint, file_prefix_feed_dict, _graph_view);
    }
}

public class CheckpointRestoreCoordinator
{
    private CheckpointOptions _options;
    private TrackableObjectGraph _object_graph_proto;
    private int _restore_uid;
    private HashSet<int> _matched_proto_ids;
    private Tensor _save_path_tensor;
    private string _save_path_string;
    private CheckpointReader _reader;
    private Dictionary<string, TF_DataType> _dtype_map;
    private Dictionary<string, Shape> _shape_map;
    private ObjectGraphView _graph_view;
    private Dictionary<int, IList<SlotVariableRestoration>> _slot_restorations;
    private bool _expect_partial_attr;
    private List<Operation> _restore_ops;
    private List<Trackable> _all_trackables;
    private Dictionary<int, Trackable> _object_by_proto_id;
    private Dictionary<string, Operation> _restore_ops_by_name;
    private Dictionary<int, IList<DeferredSlotVariableRestoration>> _deferred_slot_restorations;
    private Dictionary<int, IList<string>> _unused_attributes;

    public CheckpointRestoreCoordinator(TrackableObjectGraph object_graph_proto, string save_path, Tensor save_path_tensor, 
        CheckpointReader reader, object? restore_op_cache, ObjectGraphView graph_view, CheckpointOptions options, object? saveables_cache)
    {
        // TODO(Rinne): cache.
        _options = options;
        _object_graph_proto = object_graph_proto;
        _restore_uid = ops.uid();
        _save_path_tensor = save_path_tensor;
        _save_path_string = save_path;
        _reader = reader;
        if(_reader is null)
        {
            _reader = new CheckpointReader(save_path);
        }
        _dtype_map = _reader.VariableToDataTypeMap;
        _shape_map = _reader.VariableToShapeMap;
        _graph_view = graph_view;
        _restore_ops = new List<Operation>();
        _restore_ops_by_name = new Dictionary<string, Operation>();
        _all_trackables = new List<Trackable>();
        _matched_proto_ids = new HashSet<int>();
        _object_by_proto_id = new Dictionary<int, Trackable>();
        _slot_restorations = new Dictionary<int, IList<SlotVariableRestoration>>();
        _deferred_slot_restorations = new Dictionary<int, IList<DeferredSlotVariableRestoration>>();

        _expect_partial_attr = false;
        for(int i = 0; i < _object_graph_proto.Nodes.Count; i++)
        {
            var node = _object_graph_proto.Nodes[i];
            foreach(var slot_reference in node.SlotVariables)
            {
                _slot_restorations.SetDefault(slot_reference.OriginalVariableNodeId, new List<SlotVariableRestoration>())
                    .Add(new SlotVariableRestoration(i, slot_reference.SlotVariableNodeId, slot_reference.SlotName));
            }
        }

        // skip the deleter and cache.
    }

    public bool ExpectPartial
    {
        get
        {
            return _expect_partial_attr;
        }
        set
        {
            _expect_partial_attr = value;
        }
    }

    /// <summary>
    /// Corresponding to `all_python_objects` of tensorflow python
    /// </summary>
    public List<Trackable> AllTrackables => _all_trackables;
    public HashSet<int> MatchedProtoIds => _matched_proto_ids;
    // TODO(Rinne): change to weak ref.
    public Dictionary<int, Trackable> ObjectByProtoId => _object_by_proto_id;
    public int RestoreUid => _restore_uid;
    public TrackableObjectGraph ObjectGraphProto => _object_graph_proto;
    public Dictionary<int, IList<SlotVariableRestoration>> SlotRestorations => _slot_restorations;
    public Dictionary<int, IList<DeferredSlotVariableRestoration>> DeferredSlotRestorations => _deferred_slot_restorations;
    public Dictionary<string, Operation> RestoreOpsByName => _restore_ops_by_name;
    public Dictionary<int, IList<string>> UnusedAttributes => _unused_attributes;

    public void new_restore_ops(IEnumerable<Operation> new_ops)
    {
        _restore_ops.AddRange(new_ops);
        // skip the callback.
    }

    public List<Operation> restore_saveables(Dictionary<string, OneOf<BaseResourceVariable, MySaveableObject>> tensor_saveables, List<CheckpointPosition> positions, object? registered_savers = null)
    {
        List<Operation> restore_ops = new();
        foreach(var position in positions)
        {
            var key = position.ObjectProto.Attributes[0].CheckpointKey;
            throw new NotImplementedException();
        }

        Dictionary<string, BaseResourceVariable> variable_dict = new();
        foreach(var item in tensor_saveables)
        {
            if(item.Value.TryPickT0(out var variable, out var _))
            {
                variable_dict[item.Key] = variable;
            }
            else
            {
                throw new TypeError();
            }
        }

        if (tensor_saveables is not null && tensor_saveables.Count > 0)
        {
            var flat_saveables = saveable_object_util.validate_and_slice_inputs(variable_dict);
            var new_restore_ops = MultiDeviceSaver.from_saveables(flat_saveables).restore(_save_path_tensor, _options);
            if (!tf.Context.executing_eagerly())
            {
                foreach(var item in new_restore_ops)
                {
                    restore_ops.Add(item.Value);
                    Debug.Assert(!_restore_ops_by_name.ContainsKey(item.Key));
                    _restore_ops_by_name[item.Key] = item.Value;
                }
            }
        }
        return restore_ops;
    }
}

public abstract class LoadStatus
{
    public abstract LoadStatus assert_consumed();
    public abstract LoadStatus assert_existing_objects_matched();
    public abstract LoadStatus assert_nontrivial_match();
    public abstract LoadStatus run_restore_ops(Session? session = null);
    public abstract void initialize_or_restore(Session? session = null);
    public virtual LoadStatus expect_partial()
    {
        return this;
    }
}

public class InitializationOnlyStatus: LoadStatus
{
    private int _restore_uid;
    private ObjectGraphView _object_graph_view;
    private Trackable _root;
    public InitializationOnlyStatus(ObjectGraphView object_graph_view, int restore_uid)
    {
        _restore_uid = restore_uid;
        _object_graph_view = object_graph_view;
        _root = object_graph_view.Root;
    }
    public override LoadStatus assert_consumed()
    {
        throw new AssertionError("No checkpoint specified (save_path=None); nothing is being restored.");
    }
    public override LoadStatus assert_existing_objects_matched()
    {
        throw new AssertionError("No checkpoint specified (save_path=None); nothing is being restored.");
    }
    public override LoadStatus assert_nontrivial_match()
    {
        throw new AssertionError("No checkpoint specified (save_path=None); nothing is being restored.");
    }
    public override LoadStatus run_restore_ops(Session? session = null)
    {
        throw new AssertionError("No checkpoint specified, so no restore ops are available "
        + "(save_path=None to Saver.restore).");
    }
    public override void initialize_or_restore(Session? session = null)
    {
        if (tf.Context.executing_eagerly())
        {
            return;
        }
        if(session is null)
        {
            session = new Session();
        }
        var trackable_objects = CheckPointUtils.list_objects(_object_graph_view);
        throw new NotImplementedException("Not implemented, please submit an issue to https://github.com/SciSharp/TensorFlow.NET/issues");
    }
}

internal class CheckpointLoadStatus: LoadStatus
{
    private CheckpointRestoreCoordinator _checkpoint;
    private Dictionary<Tensor, string> _feed_dict;
    private ObjectGraphView _object_graph_view;
    private Trackable _root;
    public CheckpointLoadStatus(CheckpointRestoreCoordinator checkpoint, Dictionary<Tensor, string> feed_dict, ObjectGraphView graph_view):base()
    {
        _checkpoint = checkpoint;
        _feed_dict = feed_dict;
        _object_graph_view = graph_view;
        _root = graph_view.Root;
    }

    public CheckpointRestoreCoordinator Checkpoint => _checkpoint;

    public override LoadStatus assert_consumed()
    {
        throw new NotImplementedException();
    }

    public override LoadStatus assert_existing_objects_matched()
    {
        for(int i = 0; i < _checkpoint.ObjectGraphProto.Nodes.Count; i++)
        {
            var node = _checkpoint.ObjectGraphProto.Nodes[i];
            if(_checkpoint.ObjectByProtoId.TryGetValue(i, out var trackable) && 
                trackable.UpdateUid < _checkpoint.RestoreUid)
            {
                throw new AssertionError($"Object {node} not assigned a value from checkpoint.");
            }
        }
        foreach(var trackable_object in CheckPointUtils.list_objects(_object_graph_view))
        {
            if(trackable_object is TrackableDataStructure && trackable_object._trackable_children().Count == 0)
            {
                continue;
            }
            _checkpoint.AllTrackables.Add(trackable_object);
        }
        var unused_trackables = CheckPointUtils._objects_with_attributes(_checkpoint.AllTrackables)
            .Except(_checkpoint.ObjectByProtoId.Values);
        if (unused_trackables.Any())
        {
            var num_unused_trackables = unused_trackables.Count();
            var num_variables_to_show = Math.Min(10, num_unused_trackables);
            throw new AssertionError($"Found {num_unused_trackables} Python objects that were " +
                $"not bound to checkpointed values, likely due to changes in the " +
                $"Python program. Showing {num_variables_to_show} of " +
                $"{num_unused_trackables} unmatched objects: " +
                $"{{list(unused_python_objects)[:num_variables_to_show]}}");
        }
        return this;
    }

    public override LoadStatus assert_nontrivial_match()
    {
        throw new NotImplementedException();
    }

    public override LoadStatus expect_partial()
    {
        throw new NotImplementedException();
    }

    public override void initialize_or_restore(Session? session = null)
    {
        throw new NotImplementedException();
    }

    public override LoadStatus run_restore_ops(Session? session = null)
    {
        throw new NotImplementedException();
    }
}