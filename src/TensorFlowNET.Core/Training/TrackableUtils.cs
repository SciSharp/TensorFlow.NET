using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Checkpoint;
using Tensorflow.Exceptions;
using Tensorflow.Train;

namespace Tensorflow.Training;

public static class TrackableUtils
{
    public class CyclicDependencyError: System.Exception
    {
        public IDictionary<int, IEnumerable<int>> LeftOverDependencyMap { get; }
        public CyclicDependencyError(IDictionary<int, IEnumerable<int>> leftover_dependency_map): base()
        {
            LeftOverDependencyMap = leftover_dependency_map;
        }
        public CyclicDependencyError(IDictionary<int, List<int>> leftover_dependency_map): base()
        {
            LeftOverDependencyMap = leftover_dependency_map.ToDictionary(x => x.Key, x => x.Value.AsEnumerable());
        }
    }
    internal static string _ESCAPE_CHAR = ".";
    internal static string _OPTIMIZER_SLOTS_NAME = _ESCAPE_CHAR + "OPTIMIZER_SLOT";
    internal static string OBJECT_ATTRIBUTES_NAME = _ESCAPE_CHAR + "ATTRIBUTES";
    internal static string SERIALIZE_TO_TENSORS_NAME = _ESCAPE_CHAR + "TENSORS";
    public static string object_path_to_string(IEnumerable<TrackableReference> node_path_arr)
    {
        return string.Join("/", node_path_arr.Select(x => escape_local_name(x.Name)));
    }

    public static string escape_local_name(string name)
    {
        return name.Replace(_ESCAPE_CHAR, _ESCAPE_CHAR + _ESCAPE_CHAR).Replace("/", _ESCAPE_CHAR + "S");
    }
    
    public static string checkpoint_key(string object_path, string local_name)
    {
        var key_suffix = escape_local_name(local_name);
        if (local_name == SERIALIZE_TO_TENSORS_NAME)
        {
            key_suffix = "";
        }

        return $"{object_path}/{OBJECT_ATTRIBUTES_NAME}/{key_suffix}";
    }

    /// <summary>
    /// Topologically sorts the keys of a map so that dependencies appear first.
    /// Uses Kahn's algorithm: https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm
    /// </summary>
    /// <param name="dependency_map"></param>
    /// <exception cref="ValueError"></exception>
    public static List<int> order_by_dependency(IDictionary<int, List<int>> dependency_map)
    {
        Dictionary<int, HashSet<int>> reverse_dependency_map = new();
        foreach (var pair in dependency_map)
        {
            foreach (var dep in pair.Value)
            {
                if (reverse_dependency_map.ContainsKey(dep))
                {
                    reverse_dependency_map[dep].Add(pair.Key);
                }
                else
                {
                    reverse_dependency_map[dep] = new HashSet<int>();
                    reverse_dependency_map[dep].Add(pair.Key);
                }
            }
        }
        
        // Validate that all values in the dependency map are also keys.
        var unknown_keys = reverse_dependency_map.Keys.Except(dependency_map.Keys);
        if (unknown_keys.Count() > 0)
        {
            throw new ValueError(
                $"Found values in the dependency map which are not keys: {string.Join(", ", unknown_keys.Select(x => x.ToString()))}");
        }
        
        // Generate the list sorted by objects without dependencies -> dependencies.
        // The returned list will reverse this.
        List<int> reversed_dependency_arr = new();

        Queue<int> to_visit = new();
        foreach (var x in dependency_map.Keys)
        {
            if (!reverse_dependency_map.ContainsKey(x))
            {
                to_visit.Enqueue(x);
            }
        }

        while (to_visit.Count > 0)
        {
            var x = to_visit.Dequeue();
            reversed_dependency_arr.Add(x);
            foreach (var dep in dependency_map[x].Distinct())
            {
                var edges = reverse_dependency_map[dep];
                edges.Remove(x);
                if (edges.Count == 0)
                {
                     to_visit.Enqueue(dep);
                    if (!reverse_dependency_map.Remove(dep))
                    {
                        throw new KeyError($"Cannot find the key {dep} in reverse_dependency_map");
                    }
                }
            }
        }

        if (reverse_dependency_map.Count > 0)
        {
            Dictionary<int, List<int>> leftover_dependency_map = new();
            foreach (var pair in reverse_dependency_map)
            {
                foreach (var x in pair.Value)
                {
                    if (leftover_dependency_map.ContainsKey(x))
                    {
                        leftover_dependency_map[x].Add(pair.Key);
                    }
                    else
                    {
                        leftover_dependency_map[x] = new List<int>() { pair.Key };
                    }
                }
            }

            throw new CyclicDependencyError(leftover_dependency_map);
        }

        reversed_dependency_arr.Reverse();
        return reversed_dependency_arr;
    }

    public static string pretty_print_node_path(IEnumerable<TrackableReference> paths)
    {
        if (paths.Count() == 0)
        {
            return "root object";
        }
        else
        {
            return $"root.{string.Join(".", paths.Select(x => x.Name))}";
        }
    }

    /// <summary>
    /// Returns the substring after the "/.ATTIBUTES/" in the checkpoint key.
    /// </summary>
    /// <param name="key"></param>
    /// <param name="prefix"></param>
    /// <returns></returns>
    public static string extract_local_name(string key, string? prefix = null)
    {
        if(prefix is null)
        {
            prefix = "";
        }
        var search_key = OBJECT_ATTRIBUTES_NAME + "/" + prefix;
        try
        {
            return key.Substring(key.IndexOf(search_key) + search_key.Length);
        }
        catch(ArgumentOutOfRangeException)
        {
            return key;
        }
    }
}