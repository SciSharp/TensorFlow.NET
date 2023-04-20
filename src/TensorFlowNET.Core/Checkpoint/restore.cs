using OneOf;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Security;
using System.Text;
using Tensorflow.Train;
using Tensorflow.Training;
using static Tensorflow.Binding;

namespace Tensorflow.Checkpoint;

public class CheckpointPosition
{
    private CheckpointRestoreCoordinator _checkpoint;
    private int _proto_id;
    private bool _skip_restore;
    public CheckpointPosition(CheckpointRestoreCoordinator checkpoint, int proto_id)
    {
        _checkpoint = checkpoint;
        _proto_id = proto_id;
        _skip_restore = false;
    }

    public Trackable Trackable => _checkpoint.ObjectByProtoId[_proto_id];
    public CheckpointRestoreCoordinator Checkpoint => _checkpoint;
    public TrackableObjectGraph.Types.TrackableObject ObjectProto => _checkpoint.ObjectGraphProto.Nodes[_proto_id];

    public void restore(Trackable trackable)
    {
        using (ops.init_scope())
        {
            if (bind_project(trackable))
            {
                var restore_ops = _restore_descendants();
                if(restore_ops is not null && restore_ops.Count > 0)
                {
                    _checkpoint.new_restore_ops(restore_ops);
                }
            }
        }
    }

    /// <summary>
    /// Set a checkpoint<->object correspondence.
    /// </summary>
    /// <param name="trackable"></param>
    /// <returns></returns>
    public bool bind_project(Trackable trackable)
    {
        _checkpoint.AllTrackables.Add(trackable);
        _checkpoint.MatchedProtoIds.Add(_proto_id);
        if(_checkpoint.ObjectByProtoId.TryGetValue(_proto_id, out var current_assignment) && current_assignment is not null)
        {
            // skip the `logging.warning`.
            return false;
        }
        else
        {
            _checkpoint.ObjectByProtoId[_proto_id] = trackable;
            return true;
        }
    }

    public (List<Operation>, Dictionary<string, OneOf<BaseResourceVariable, MySaveableObject>>, List<CheckpointPosition>, object?) gather_ops_or_named_saveables()
    {
        // skip the registered_saver

        if (ObjectProto.Attributes is null || ObjectProto.Attributes.Count == 0)
        {
            return (new List<Operation>(), new Dictionary<string, OneOf<BaseResourceVariable, MySaveableObject>>(),
                new List<CheckpointPosition>(), null);
        }

        var saveable_factories = saveable_object_util.saveable_objects_from_trackable(this.Trackable);

        List<Operation> existing_restore_ops;
        List<CheckpointPosition> positions = new();
        Dictionary<string, OneOf<BaseResourceVariable, MySaveableObject>> named_saveables;
        if (saveable_factories.Keys.Count == 1 && saveable_factories.Keys.First() == TrackableUtils.SERIALIZE_TO_TENSORS_NAME)
        {
            (existing_restore_ops, named_saveables) = _create_serialize_to_tensor_saveable(saveable_factories);
        }
        else if(saveable_factories.Count > 0)
        {
            (existing_restore_ops, named_saveables) = _create_saveables_by_attribute_name(saveable_factories);
        }
        else
        {
            throw new NotImplementedException();
        }
        return (existing_restore_ops, named_saveables, positions, null);
    }

    public CheckpointPosition create_child_position(int node_id)
    {
        return new CheckpointPosition(_checkpoint, node_id);
    }

    public (CheckpointPosition, BaseResourceVariable) create_slot_variable_position(Optimizer optimizer_object, BaseResourceVariable variable, 
        int slot_variable_id, string slot_name)
    {
        //CheckpointPosition slot_variable_position = new(Checkpoint, slot_variable_id);

        // TODO(Rinne): implement it.
        return (null, null);
    }

    /// <summary>
    /// Creates a saveable using the _serialize_to_tensor method.
    /// </summary>
    /// <param name="saveable_factories"></param>
    private (List<Operation>, Dictionary<string, OneOf<BaseResourceVariable, MySaveableObject>>) _create_serialize_to_tensor_saveable(
        IDictionary<string, Func<string, OneOf<BaseResourceVariable, MySaveableObject>>> saveable_factories)
    {
        string suffix = SaveableCompat.get_saveable_name(this.Trackable);
        suffix = suffix ?? "";
        var saveable_name = _extract_saveable_name(ObjectProto.Attributes[0].CheckpointKey) + suffix;

        if (!tf.Context.executing_eagerly())
        {
            throw new NotImplementedException("The restore under graph mode has not been implemented. " +
                "Please submit an issue to https://github.com/SciSharp/TensorFlow.NET/issues");
        }

        var saveable = saveable_factories[TrackableUtils.SERIALIZE_TO_TENSORS_NAME](saveable_name);
        // skip the cache.
        Dictionary<string, OneOf<BaseResourceVariable, MySaveableObject>> dict = new();
        dict[saveable_name] = saveable;
        return (new List<Operation>(), dict);
    }

    private (List<Operation>, Dictionary<string, OneOf<BaseResourceVariable, MySaveableObject>>) _create_saveables_by_attribute_name(
        IDictionary<string, Func<string, OneOf<BaseResourceVariable, MySaveableObject>>> saveable_factories)
    {
        // TODO(Rinne): implement it.
        if(ObjectProto.Attributes is null)
        {
            return (new List<Operation>(), new Dictionary<string, OneOf<BaseResourceVariable, MySaveableObject>>());
        }

        List<Operation> existing_restore_ops = new();
        HashSet<string> created_compat_names = new();
        Dictionary<string, OneOf<BaseResourceVariable, MySaveableObject>> named_saveables = new();
        foreach (var serialized_tensor in ObjectProto.Attributes)
        {
            Operation existing_op;
            if (tf.Context.executing_eagerly() || !_checkpoint.RestoreOpsByName.ContainsKey(serialized_tensor.CheckpointKey))
            {
                existing_op = null;
            }
            else
            {
                existing_op = _checkpoint.RestoreOpsByName[serialized_tensor.CheckpointKey];
            }

            if(existing_op is not null)
            {
                existing_restore_ops.Add(existing_op);
                continue;
            }

            if(created_compat_names.Any(x => serialized_tensor.Name.StartsWith(x)))
            {
                continue;
            }

            // TODO(Rinne): deal with cache.

            var saveable = _get_saveable_from_factory(saveable_factories, serialized_tensor, created_compat_names);
            if(saveable is null)
            {
                _checkpoint.UnusedAttributes.SetDefault(_proto_id, new List<string>()).Add(serialized_tensor.Name);
                continue;
            }
            named_saveables[serialized_tensor.CheckpointKey] = saveable.Value;
        }
        return (existing_restore_ops, named_saveables);
    }

    private OneOf<BaseResourceVariable, MySaveableObject>? _get_saveable_from_factory(IDictionary<string, Func<string, OneOf<BaseResourceVariable, MySaveableObject>>> saveable_factories,
        TrackableObjectGraph.Types.TrackableObject.Types.SerializedTensor serialized_tensor, HashSet<string> created_compat_names)
    {
        var expected_factory_name = serialized_tensor.Name;
        var factory_input_name = serialized_tensor.CheckpointKey;

        if (!saveable_factories.TryGetValue(expected_factory_name, out var matched_factory))
        {
            foreach(var item in saveable_factories)
            {
                var factory_name = item.Key;
                var factory = item.Value;
                if (expected_factory_name.StartsWith(factory_name))
                {
                    if(matched_factory is not null)
                    {
                        throw new ValueError($"Forward compatibility load error: Unable to load " + 
                           "checkpoint saved in future version of TensorFlow. " + 
                           "Please update your version of TensorFlow to the " + 
                           "version in which the checkpoint was saved.");
                    }
                }
                matched_factory = factory;
                factory_input_name = _extract_saveable_name(serialized_tensor.CheckpointKey) + factory_name;
                created_compat_names.Add(factory_name);
            }
        }
        return matched_factory(factory_input_name);
    }

    private string _extract_saveable_name(string checkpoint_key)
    {
        var search_key = TrackableUtils.OBJECT_ATTRIBUTES_NAME + "/";
        return checkpoint_key.Substring(0, checkpoint_key.IndexOf(search_key) + search_key.Length);
    }

    /// <summary>
    /// Restore the bound Trackable and dependencies (may be deferred).
    /// </summary>
    private List<Operation> _restore_descendants()
    {
        Queue<(CheckpointPosition, Trackable)> visit_queue = new();
        visit_queue.Enqueue((this, this.Trackable));
        List<Operation> restore_ops = new();
        Dictionary<string, OneOf<BaseResourceVariable, MySaveableObject>> tensor_saveables = new();
        List<CheckpointPosition> positions = new();

        CheckpointPosition current_position = null;
        while (visit_queue.Count > 0)
        {
            current_position = visit_queue.Dequeue().Item1;
            var (new_restore_ops, new_tensor_saveables, new_positions, new_registered_savers) = current_position._single_restore();
            restore_ops.AddRange(new_restore_ops);
            foreach(var item in new_tensor_saveables)
            {
                tensor_saveables.Add(item.Key, item.Value);
            }
            positions.AddRange(new_positions);
            _queue_children_for_restoration(current_position, visit_queue);
            _queue_slot_variables(current_position, visit_queue);
        }
        restore_ops.AddRange(current_position.Checkpoint.restore_saveables(tensor_saveables, positions, null));
        return restore_ops;
    }

    private void _queue_children_for_restoration(CheckpointPosition checkpoint_position, Queue<(CheckpointPosition, Trackable)> visit_queue)
    {
        var trackable = checkpoint_position.Trackable;
        foreach(var child in checkpoint_position.ObjectProto.Children)
        {
            var child_position = checkpoint_position.create_child_position(child.NodeId);
            var local_object = trackable._lookup_dependency(child.LocalName);
            var child_proto = child_position.ObjectProto;
            if(local_object is null)
            {
                if(child_proto.Children.Any() || child_proto.Attributes.Any() || child_proto.SlotVariables.Any())
                {
                    trackable.DeferredDependencies.SetDefault(child.LocalName, new List<CheckpointPosition>()).Add(child_position);
                }
            }
            else
            {
                if (child_position.bind_project(local_object))
                {
                    visit_queue.Enqueue((child_position, local_object));
                }
            }
        }
    }

    private void _queue_slot_variables(CheckpointPosition checkpoint_position, Queue<(CheckpointPosition, Trackable)> visit_queue)
    {
        var trackable = checkpoint_position.Trackable;
        var checkpoint = checkpoint_position.Checkpoint;
        if(checkpoint.DeferredSlotRestorations.TryGetValue(checkpoint_position._proto_id, out var positions))
        {
            checkpoint.DeferredSlotRestorations.Remove(checkpoint_position._proto_id);
            foreach (var deferred_slot_restoration in positions)
            {
                var (slot_variable_position, slot_variable) = checkpoint_position.create_slot_variable_position(
                    trackable as Optimizer, deferred_slot_restoration.OriginalVariable, deferred_slot_restoration.SlotVariableId,
                    deferred_slot_restoration.SlotName
                );
                if(slot_variable_position is not null)
                {
                    visit_queue.Enqueue((slot_variable_position, slot_variable));
                }
            }
        }
        if (checkpoint.SlotRestorations.TryGetValue(checkpoint_position._proto_id, out var restorations))
        {
            checkpoint.SlotRestorations.Remove(checkpoint_position._proto_id);
            foreach (var slot_restoration in restorations)
            {
                if(Checkpoint.ObjectByProtoId.TryGetValue(slot_restoration.OptimizerId, out var optimizer_object))
                {
                    throw new NotImplementedException();
                    // TODO(Rinne)； implement it.
                }
                else
                {
                    Debug.Assert(trackable is BaseResourceVariable);
                    Checkpoint.DeferredSlotRestorations.SetDefault(slot_restoration.OptimizerId, new List<DeferredSlotVariableRestoration>())
                        .Add(new DeferredSlotVariableRestoration(trackable as BaseResourceVariable, slot_restoration.SlotVariableId, slot_restoration.SlotName));
                }
            }
        }
    }

    private (List<Operation>, Dictionary<string, OneOf<BaseResourceVariable, MySaveableObject>>, List<CheckpointPosition>, object?) _single_restore()
    {
        var trackable = this.Trackable;
        trackable._maybe_initialize_trackable();
        if(_checkpoint.RestoreUid > trackable.UpdateUid)
        {
            var (restore_ops, tensor_saveables, positions, registered_savers) = gather_ops_or_named_saveables();
            trackable.UpdateUid = _checkpoint.RestoreUid;
            return (restore_ops, tensor_saveables, positions, registered_savers);
        }
        else
        {
            return (new List<Operation>(), new Dictionary<string, OneOf<BaseResourceVariable, MySaveableObject>>(),
                new List<CheckpointPosition>(), null);
        }
    }
}

public record class DeferredSlotVariableRestoration(
    BaseResourceVariable OriginalVariable, 
    int SlotVariableId,
    string SlotName
);