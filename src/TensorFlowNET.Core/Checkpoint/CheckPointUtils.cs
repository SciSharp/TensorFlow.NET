using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Tensorflow.Functions;
using Tensorflow.Train;
using Tensorflow.Training;
using pbc = global::Google.Protobuf.Collections;

namespace Tensorflow.Checkpoint;

public static class CheckPointUtils
{
    private static string _ESCAPE_CHAR = ".";
    public static (IList<Trackable>, IDictionary<Trackable, IEnumerable<TrackableReference>>, IDictionary<Trackable, int>,
        IDictionary<Trackable, pbc::RepeatedField<TrackableObjectGraph.Types.TrackableObject.Types.SlotVariableReference>>,
        IDictionary<Trackable, string>) objects_ids_and_slot_variables_and_paths(ObjectGraphView graph_view)
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

        var slot_variables = serialize_slot_variables(trackable_objects, node_ids, object_names);
        return (trackable_objects, node_paths, node_ids, slot_variables, object_names);
    }

    public static
        IDictionary<Trackable, pbc::RepeatedField<global::Tensorflow.TrackableObjectGraph.Types.TrackableObject.Types.SlotVariableReference>>
        serialize_slot_variables(IEnumerable<Trackable> trackable_objects,
            IDictionary<Trackable, int> node_ids, IDictionary<Trackable, string> object_names)
    {
        var non_slot_objects = trackable_objects.ToList();
        Dictionary<Trackable, pbc::RepeatedField<global::Tensorflow.TrackableObjectGraph.Types.TrackableObject.Types.SlotVariableReference>>
            slot_variables = new();
        foreach (var trackable in non_slot_objects)
        {
            if (trackable is not Optimizer)
            {
                continue;
            }

            var optim = (Optimizer)trackable;
            var slot_names = optim.get_slot_names();
            foreach (var slot_name in slot_names)
            {
                for (int original_variable_node_id = 0;
                     original_variable_node_id < non_slot_objects.Count;
                     original_variable_node_id++)
                {
                    var original_variable = non_slot_objects[original_variable_node_id];
                    IVariableV1 slot_variable;
                    if (original_variable is not IVariableV1)
                    {
                        slot_variable = null;
                    }
                    slot_variable = optim.get_slot((IVariableV1)original_variable, slot_name);
                    if(slot_variable is null) continue;

                    // There're some problems about the inherits of `Variable` and `Trackable`.
                    throw new NotImplementedException();
                }
            }
        }

        return slot_variables;
    }

    public static Trackable get_mapped_trackable(Trackable trackable, IDictionary<Trackable, Trackable>? object_map)
    {
        if (object_map is null || !object_map.TryGetValue(trackable, out var possible_res))
        {
            return trackable;
        }
        else
        {
            return possible_res;
        }
    }

    public static string get_full_name(Trackable variable)
    {
        // TODO: This state is not correct, the whole framework need to be updated in the future.
        if (!(variable is IVariableV1 || resource_variable_ops.is_resource_variable(variable)))
        {
            return "";
        }
        // skip the check of attribute `_save_slice_info` .

        // TODO: Need to be revised!!!
        Debug.Assert(variable is BaseResourceVariable);
        return ((BaseResourceVariable)variable).Name;
    }

    public static void add_checkpoint_values_check(TrackableObjectGraph object_graph_proto)
    {
        HashSet<int> checkpointed_trackables = new();
        Dictionary<int, HashSet<int>> parents = new();
        for (int i = 0; i < object_graph_proto.Nodes.Count; i++)
        {
            var object_proto = object_graph_proto.Nodes[i];
            // skip the process of registered saver.
            if (object_proto.Attributes is not null && object_proto.Attributes.Count > 0 ||
                object_proto.SlotVariables is not null && object_proto.SlotVariables.Count > 0)
            {
                checkpointed_trackables.Add(i);
            }

            foreach (var child_proto in object_proto.Children)
            {
                var child = child_proto.NodeId;
                if (!parents.ContainsKey(child))
                {
                    parents[child] = new HashSet<int>();
                }

                parents[child].Add(i);
            }
        }

        Queue<int> to_visit = new(checkpointed_trackables.AsEnumerable());
        while (to_visit.Count > 0)
        {
            var trackable = to_visit.Dequeue();
            if (!parents.ContainsKey(trackable)) continue;
            var current_parents = parents[trackable];
            foreach (var parent in current_parents)
            {
                checkpointed_trackables.Add(parent);
                if (parents.ContainsKey(parent))
                {
                    to_visit.Enqueue(parent);
                }
            }
            parents.Remove(trackable);
        }
        
        // TODO: Complete it after supporting checkpoint.
        // for (int i = 0; i < object_graph_proto.Nodes.Count; i++)
        // {
        //     object_graph_proto.Nodes[i].has_checkpoint_values.value = checkpointed_trackables.Contains(i);
        // }
    }

    /// <summary>
    /// Traverse the object graph and list all accessible objects.
    /// </summary>
    /// <param name="object_graph_view"></param>
    public static IList<Trackable> list_objects(ObjectGraphView graph_view)
    {
        return objects_ids_and_slot_variables_and_paths(graph_view).Item1;
    }

    internal static IEnumerable<Trackable> _objects_with_attributes(IEnumerable<Trackable> full_list)
    {
        return full_list.Where(x =>
        {
            var saveables = x.gather_saveables_for_checkpoint();
            return saveables is not null && saveables.Count > 0;
        });
    }
}
