using OneOf;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using Tensorflow.Train;
using Tensorflow.Training;
using Tensorflow.Common.Extensions;
using pbc = global::Google.Protobuf.Collections;

namespace Tensorflow.Checkpoint
{
    internal record class TrackableData(
        // A trackable in the root Trackable object graph.
        Trackable trackable,
        // The index at which the Trackable appears in TrackableObjectGraph.nodes.
        int node_id,
        // The BFS-generated path from the root object / used to generate readable checkpoint keys.
        string object_name,
        // A list of ObjectReference for each child connected to this Trackable.
        pbc::RepeatedField<global::Tensorflow.TrackableObjectGraph.Types.TrackableObject.Types.ObjectReference> children_proto,
        // A list of SlotVariableReference to save to the object (only valid for Optimizer objects).
        pbc::RepeatedField<global::Tensorflow.TrackableObjectGraph.Types.TrackableObject.Types.SlotVariableReference> slot_variable_proto,
        // The object to save to checkpoint. Usually this is the same as `trackable`,
        // but can differ when the the caller wants to specify a different object to
        // save. For example, when saving checkpoints asynchronously, variables are
        // copied to the CPU. `object_to_save` is set as the copied variable.
        Trackable object_to_save
    );
    public static class SaveUtil
    {
        public static (IDictionary<Trackable, IDictionary<string, IDictionary<string, OneOf<Tensor, SaveSpec>>>>, IDictionary<Tensor, object>, IDictionary<string, IDictionary<string, Trackable>>, TrackableObjectGraph) 
            serialize_graph_view(ObjectGraphView graph_view, IDictionary<Trackable, Trackable>? object_map = null, bool call_with_mapped_captures = false, object? cache = null)
        {
            var (trackable_data, node_ids) = gather_trackable_data(graph_view, object_map);
            var (tensor_trackables, pystate_trackables, registered_trackables) = split_trackables(trackable_data);

            var object_graph_proto = fill_object_graph_proto(trackable_data);

            var serialized_tensors = get_and_write_tensors_to_serialize(tensor_trackables, node_ids, call_with_mapped_captures, cache, object_graph_proto);
            var registered_savers = get_and_write_registered_savers(registered_trackables, object_graph_proto);

            Dictionary<Tensor, object> feed_additions;
            if(cache is null)
            {
                feed_additions = null;
                serialized_tensors = serialized_tensors.Concat(get_and_write_tensors_to_serialize(pystate_trackables, node_ids, call_with_mapped_captures,
                    cache, object_graph_proto)).ToDictionary(x => x.Key, x => x.Value);
            }
            else
            {
                feed_additions = null;
                // TODO: deal with cache.
                throw new NotFiniteNumberException();
            }

            CheckPointUtils.add_checkpoint_values_check(object_graph_proto);

            return (serialized_tensors, feed_additions, registered_savers, object_graph_proto);
        }

        private static (IList<TrackableData>, IDictionary<Trackable, int>) gather_trackable_data(ObjectGraphView graph_view, IDictionary<Trackable, Trackable>? object_map)
        {
            var (trackable_objects, node_paths) = graph_view.breadth_first_traversal();
            Dictionary<Trackable, string> object_names = new();
            foreach(var pair in node_paths)
            {
                object_names[pair.Key] = TrackableUtils.object_path_to_string(pair.Value);
            }
            Dictionary<Trackable, int> node_ids = new();
            for(int i = 0; i < trackable_objects.Count; i++)
            {
                node_ids[trackable_objects[i]] = i;
            }
            var slot_variables = CheckPointUtils.serialize_slot_variables(trackable_objects, node_ids, object_names);
            List<TrackableData> trackable_data = new();
            foreach(var trackable in trackable_objects)
            {
                pbc::RepeatedField<global::Tensorflow.TrackableObjectGraph.Types.TrackableObject.Types.ObjectReference> children_proto = new();
                foreach(var child in graph_view.list_children(trackable))
                {
                    children_proto.Add(new TrackableObjectGraph.Types.TrackableObject.Types.ObjectReference()
                    {
                        NodeId = node_ids[child.Refer],
                        LocalName = child.Name
                    });
                }
                slot_variables.TryGetValue(trackable, out var slot_variable);
                trackable_data.Add(new TrackableData(
                    trackable: trackable,
                    node_id: node_ids[trackable],
                    object_name: object_names[trackable],
                    children_proto: children_proto,
                    slot_variable_proto: slot_variable??new pbc.RepeatedField<TrackableObjectGraph.Types.TrackableObject.Types.SlotVariableReference>(),
                    object_to_save: CheckPointUtils.get_mapped_trackable(trackable, object_map)
                ));
            }
            return (trackable_data, node_ids);
        }

        private static TrackableObjectGraph fill_object_graph_proto(IList<TrackableData> trackable_data)
        {
            TrackableObjectGraph object_graph_proto = new();
            for(int i = 0; i < trackable_data.Count; i++)
            {
                var td = trackable_data[i];
                Debug.Assert(td.node_id == i);
                TrackableObjectGraph.Types.TrackableObject trackable_object = new();
                trackable_object.SlotVariables.AddRange(td.slot_variable_proto);
                trackable_object.Children.AddRange(td.children_proto);
                object_graph_proto.Nodes.Add(trackable_object);
            }
            return object_graph_proto;
        }

        /// <summary>
        /// Creates dictionary of tensors to checkpoint, and updates the proto.
        /// </summary>
        /// <param name="tensor_trackables"></param>
        /// <param name="node_ids"></param>
        /// <param name="call_with_mapped_captures"></param>
        /// <param name="cache"></param>
        /// <param name="object_graph_proto"></param>
        private static IDictionary<Trackable, IDictionary<string, IDictionary<string, OneOf<Tensor, SaveSpec>>>> get_and_write_tensors_to_serialize(IList<TrackableData> tensor_trackables, IDictionary<Trackable, int> node_ids,
            bool call_with_mapped_captures, object? cache, TrackableObjectGraph object_graph_proto)
        {
            Dictionary<Trackable, IDictionary<string, IDictionary<string, OneOf<Tensor, SaveSpec>>>> serialized_tensors = new();
            foreach(var td in tensor_trackables)
            {
                // TODO: deal with cache.
                var legacy_name = SaveableCompat.get_saveable_name(td.object_to_save) ?? "";
                Trackable trackable = null;
                IDictionary<string, IDictionary<string, OneOf<Tensor, SaveSpec>>> tensor_dict;
                if(!saveable_object_util.trackable_has_serialize_to_tensor(td.object_to_save) || legacy_name.Length > 0)
                {
                    (trackable, tensor_dict) = get_tensors_from_legacy_saveable(td, node_ids, call_with_mapped_captures, object_graph_proto);
                }
                else
                {
                    tensor_dict = get_tensors_from_trackable(td, call_with_mapped_captures, object_graph_proto);
                    trackable = td.object_to_save;
                }
                if(trackable is not null)
                {
                    serialized_tensors[trackable] = tensor_dict;
                }
                else
                {
                    serialized_tensors[Trackable.None] = tensor_dict;
                }
            }
            return serialized_tensors;
        }

        private static IDictionary<string, IDictionary<string, OneOf<Tensor, SaveSpec>>> get_tensors_from_trackable(TrackableData trackable_data, bool call_with_mapped_captures, TrackableObjectGraph object_graph_proto)
        {
            var trackable = trackable_data.object_to_save;

            // TODO: complete it. Note that actually `call_with_mapped_captures` is of function type.
            IDictionary<string, IDictionary<string, OneOf<Tensor, SaveSpec>>> ret_tensor_dict;
            if (call_with_mapped_captures)
            {
                throw new NotImplementedException();
            }
            else
            {
                ret_tensor_dict = trackable.serialize_to_tensors();
            }

            Dictionary<string, IDictionary<string, OneOf<Tensor, SaveSpec>>> tensor_dict = new();
            foreach(var pair in ret_tensor_dict)
            {
                var local_name = TrackableUtils.escape_local_name(pair.Key);
                var maybe_tensor = pair.Value;
                var checkpoint_key = TrackableUtils.checkpoint_key(trackable_data.object_name, local_name);

                tensor_dict[checkpoint_key] = maybe_tensor;

                foreach(var key in maybe_tensor.Keys)
                {
                    if (maybe_tensor[key].IsTypeOrDeriveFrom<SaveSpec>())
                    {
                        maybe_tensor[key].AsT1.name = local_name + maybe_tensor[key].AsT1.name;
                    }
                }

                if(object_graph_proto is not null)
                {
                    object_graph_proto.Nodes[trackable_data.node_id].Attributes.Add(new TrackableObjectGraph.Types.TrackableObject.Types.SerializedTensor()
                    {
                        Name = local_name,
                        CheckpointKey = checkpoint_key,
                        FullName = CheckPointUtils.get_full_name(trackable)
                    });
                }
            }
            return tensor_dict;
        }

        /// <summary>
        /// Gets tensors to serialize from a Trackable with legacy SaveableObjects.
        /// </summary>
        /// <param name="trackable_data"></param>
        /// <param name="node_ids"></param>
        /// <param name="call_with_mapped_captures"></param>
        /// <param name="object_graph_proto"></param>
        /// <returns></returns>
        private static (Trackable, IDictionary<string, IDictionary<string, OneOf<Tensor, SaveSpec>>>) get_tensors_from_legacy_saveable(TrackableData trackable_data, IDictionary<Trackable, int> node_ids, 
            bool call_with_mapped_captures, TrackableObjectGraph object_graph_proto)
        {
            Dictionary<Trackable, string> object_names = new();
            object_names[trackable_data.trackable] = trackable_data.object_name;
            Dictionary<Trackable, Trackable> object_map = new();
            object_map[trackable_data.trackable] = trackable_data.object_to_save;

            var (checkpoint_factory_map, _) = SaveUtilV1.get_checkpoint_factories_and_keys(object_names, object_map);
            var (named_saveable_objects, _) = SaveUtilV1.generate_saveable_objects(checkpoint_factory_map, object_graph_proto, node_ids, object_map,
                call_with_mapped_captures, saveables_cache: null);
            var trackable = new SaveableCompatibilityConverter(trackable_data.object_to_save, named_saveable_objects);
            return (trackable, trackable.serialize_to_tensors());
        }

        private static IDictionary<string, IDictionary<string, Trackable>> get_and_write_registered_savers(IDictionary<string, IList<TrackableData>> registered_trackables, TrackableObjectGraph object_graph_proto)
        {
            Dictionary<string, IDictionary<string, Trackable>> registered_savers = new();
            foreach(var pair in registered_trackables)
            {
                foreach(var td in pair.Value)
                {
                    if (registered_savers.ContainsKey(pair.Key))
                    {
                        registered_savers[pair.Key] = new Dictionary<string, Trackable>();
                    }
                    else
                    {
                        registered_savers[pair.Key][td.object_name] = td.object_to_save;
                    }

                    var object_proto = object_graph_proto.Nodes[td.node_id];
                    // TODO: add APIs and complete it. Now the `TrackableObjectGraph.Types.TrackableObject` lacks `registered_savers`.
                }
            }
            return registered_savers;
        }

        private static (IList<TrackableData>, IList<TrackableData>, IDictionary<string, IList<TrackableData>>) split_trackables(IEnumerable<TrackableData> trackable_data)
        {
            List<TrackableData> tensor_trackables = new();
            List<TrackableData> py_state_trackables = new(); // skip the process of `PyState` for the lack of API. This is only a pleceholder.
            Dictionary<string, IList<TrackableData>> registered_trackables = new();

            foreach(var td in trackable_data)
            {
                // TODO: deal with registration.
                tensor_trackables.Add(td);
            }
            return (tensor_trackables, py_state_trackables, registered_trackables);
        }
    }
}
