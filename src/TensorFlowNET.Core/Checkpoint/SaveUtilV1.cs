using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Tensorflow.Exceptions;
using Tensorflow.Train;
using Tensorflow.Training;
using pbc = global::Google.Protobuf.Collections;
using static Tensorflow.Binding;
using Google.Protobuf;
using OneOf;

namespace Tensorflow.Checkpoint;

public static class SaveUtilV1
{
    public static (IDictionary<Trackable, IEnumerable<CheckpointFactoryData>>, object?) get_checkpoint_factories_and_keys(IDictionary<Trackable, string> object_names,
        IDictionary<Trackable, Trackable>? object_map = null)
    {
        // According to https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/registration/README.md,
        // till now only internal registrations are allowed. So, we won't return a saver in this function.
        // The implementation of this function should be updated if tensorflow update it.
        Dictionary<Trackable, IEnumerable<CheckpointFactoryData>> checkpoint_factory_map = new();
        foreach (var pair in object_names)
        {
            var trackable = pair.Key;
            var object_name = pair.Value;
            var object_to_save = CheckPointUtils.get_mapped_trackable(trackable, object_map);
            
            // skip the registration process.

            List<CheckpointFactoryData> current_list = new();
            foreach (var name_and_factory in saveable_object_util.saveable_objects_from_trackable(object_to_save))
            {
                // treat name as key_suffix.
                var name = name_and_factory.Key;
                var checkpoint_key = TrackableUtils.checkpoint_key(object_name, name);
                
                current_list.Add(new CheckpointFactoryData(name_and_factory.Value, name, checkpoint_key));
            }

            checkpoint_factory_map[trackable] = current_list;
        }

        return (checkpoint_factory_map, null);
    }

    public static (IList<MySaveableObject>, IDictionary<string, IDictionary<string, Trackable>>?) frozen_saveables_and_savers(ObjectGraphView graph_view,
        IDictionary<Trackable, Trackable> object_map, Graph? to_graph, bool call_with_mapped_captures,
        object? saveables_cache = null)
    {
        if (to_graph is not null)
        {
            var g = to_graph.as_default();
            var (named_saveable_objects, graph_proto, _, registered_savers) = serialize_gathered_objects(graph_view,
                    object_map, call_with_mapped_captures, saveables_cache);
            var object_graph_tensor = tf_with(ops.device("/cpu:0"), _ =>
            {
                // TODO(Rinne): locate the error that causes transferring TF_STRING to this function throws an exception.
                return constant_op.constant(graph_proto.ToByteArray());
            });
            named_saveable_objects.Add(new NoRestoreSaveable(object_graph_tensor, Trackable.Constants.OBJECT_GRAPH_PROTO_KEY));
            g.Exit();
            return (named_saveable_objects, registered_savers);
        }
        else
        {
            using (new ops.NullContextManager())
            {
                var (named_saveable_objects, graph_proto, _, registered_savers) = serialize_gathered_objects(graph_view,
                    object_map, call_with_mapped_captures, saveables_cache);
                var object_graph_tensor = tf_with(ops.device("/cpu:0"), _ =>
                {
                    return constant_op.constant(graph_proto.ToString());
                });
                named_saveable_objects.Add(new NoRestoreSaveable(object_graph_tensor, Trackable.Constants.OBJECT_GRAPH_PROTO_KEY));
                return (named_saveable_objects, registered_savers);
            }
        }
    }

    public static (IList<MySaveableObject>, TrackableObjectGraph, object?, IDictionary<string, IDictionary<string, Trackable>>?) serialize_gathered_objects(ObjectGraphView graph_view,
        IDictionary<Trackable, Trackable> object_map, bool call_with_mapped_captures, object? saveables_cache = null)
    {
        var (trackable_objects, node_paths) = graph_view.breadth_first_traversal();
        Dictionary<Trackable, string> object_names = new();
        foreach (var pair in node_paths)
        {
            object_names[pair.Key] = TrackableUtils.object_path_to_string(pair.Value);
        }

        Dictionary<Trackable, int> node_ids = new();
        for (int i = 0; i < trackable_objects.Count; i++)
        {
            node_ids[trackable_objects[i]] = i;
        }

        var slot_variables = CheckPointUtils.serialize_slot_variables(trackable_objects, node_ids, object_names);
        var object_graph_proto = fill_object_graph_proto(graph_view, trackable_objects, node_ids, slot_variables);
        var (named_saveable_objects, feed_additions, registered_savers) = add_attributes_to_object_graph(
            trackable_objects, object_graph_proto, node_ids, object_names, object_map, call_with_mapped_captures,
            saveables_cache);
        
        CheckPointUtils.add_checkpoint_values_check(object_graph_proto);
        return (named_saveable_objects, object_graph_proto, feed_additions, registered_savers);
    }

    private static TrackableObjectGraph fill_object_graph_proto(ObjectGraphView graph_view, IList<Trackable> trackable_objects,
        IDictionary<Trackable, int> node_ids,
        IDictionary<Trackable, pbc::RepeatedField<global::Tensorflow.TrackableObjectGraph.Types.TrackableObject.Types.SlotVariableReference>>
            slot_variables)
    {
        TrackableObjectGraph object_graph_proto = new();
        for (int i = 0; i < trackable_objects.Count; i++)
        {
            var trackable = trackable_objects[i];
            Debug.Assert(node_ids[trackable] == i);
            var object_proto = new TrackableObjectGraph.Types.TrackableObject();
            if (slot_variables.TryGetValue(trackable, out var slots))
            {
                object_proto.SlotVariables.AddRange(slots);
            }
            object_graph_proto.Nodes.Add(object_proto);
            foreach (var child in graph_view.list_children(trackable))
            {
                object_proto.Children.Add(new TrackableObjectGraph.Types.TrackableObject.Types.ObjectReference()
                    { NodeId = node_ids[child.Refer], LocalName = child.Name });
            }
        }

        return object_graph_proto;
    }

    private static (IList<MySaveableObject>, object?, IDictionary<string, IDictionary<string, Trackable>>?) add_attributes_to_object_graph(
        IList<Trackable> trackable_objects,
        TrackableObjectGraph object_graph_proto, IDictionary<Trackable, int> node_ids,
        IDictionary<Trackable, string> object_names, IDictionary<Trackable, Trackable> object_map,
        bool call_with_mapped_captures, object? saveables_cache = null)
    {
        int cnt = Math.Min(trackable_objects.Count, object_graph_proto.Nodes.Count);
        for (int i = 0; i < cnt; i++)
        {
            Debug.Assert(node_ids[trackable_objects[i]] == i);
        }

        var (checkpoint_factory_map, unmmaped_registered_savers) =
            get_checkpoint_factories_and_keys(object_names, object_map);
        
        // skip the process of registered savers

        var (named_saveable_objects, feed_additions) = generate_saveable_objects(checkpoint_factory_map,
            object_graph_proto, node_ids, object_map, call_with_mapped_captures, saveables_cache);
        return (named_saveable_objects, feed_additions, null);
    }

    public static (IList<MySaveableObject>, object?) generate_saveable_objects(
        IDictionary<Trackable, IEnumerable<CheckpointFactoryData>> checkpoint_factory_map,
        TrackableObjectGraph? object_graph_proto, IDictionary<Trackable, int>? node_ids,
        IDictionary<Trackable, Trackable> object_map, bool call_with_mapped_captures, object? saveables_cache = null)
    {
        List<MySaveableObject> named_saveable_objects = new();
        foreach (var pair in checkpoint_factory_map)
        {
            var trackable = pair.Key;
            var factory_data_list = pair.Value;
            bool fill_object_proto = object_graph_proto is not null && node_ids is not null;
            TrackableObjectGraph.Types.TrackableObject object_proto = null!;
            if (fill_object_proto)
            {
                object_proto = object_graph_proto.Nodes[node_ids[trackable]];
            }

            var object_to_save = CheckPointUtils.get_mapped_trackable(trackable, object_map);
            // skip cache

            foreach (var factory_data in factory_data_list)
            {
                var name = factory_data.name;
                var key = factory_data.checkpoint_key;
                var maybe_saveable = saveable_object_util.create_saveable_object(name, key, factory_data.factory);

                // TODO: tensorflow python has a process with callable `saveable_factory`.
                List<MySaveableObject> saveables = new();
                if (maybe_saveable.TryPickT1(out var s, out var variable))
                {
                    saveables.Add(s);
                }
                else
                {
                    saveables.AddRange(saveable_object_util.saveable_objects_for_op(variable as Trackable, key));
                }

                foreach (var saveable in saveables)
                {
                    if (!saveable.name.Contains(key))
                    {
                        throw new AssertionError($"The object {trackable} produced a SaveableObject with name " +
                                                 $"'{saveable.name}' for attribute '{name}'. Expected a name" +
                                                 $" containing '{key}'.");
                    }
                }
                
                // skip the process of PythonState
                
                named_saveable_objects.AddRange(saveables);
                
                if(!fill_object_proto) continue;

                // skip the process of `TrackableSaveable` because of lack of APIs.

                object_proto!.Attributes.Add(new TrackableObjectGraph.Types.TrackableObject.Types.SerializedTensor()
                    { Name = name, CheckpointKey = key, FullName = CheckPointUtils.get_full_name(object_to_save) });
            }
        }

        return (named_saveable_objects, null);
    }
}

public record class CheckpointFactoryData
(
    Func<string, OneOf<BaseResourceVariable, MySaveableObject>> factory,
    string name,
    string checkpoint_key
);
