using System;
using Tensorflow.Train;
using System.Collections.Generic;
using System.IO;

namespace Tensorflow.Checkpoint;

public class TrackableView
{
    protected WeakReference<Trackable> _root_ref;
    public TrackableView(Trackable obj)
    {
        _root_ref = new WeakReference<Trackable>(obj);
    }

    public TrackableView(WeakReference<Trackable> obj)
    {
        _root_ref = obj;
    }
    
    public virtual IDictionary<string, Trackable> children(Trackable obj, SaveType save_type = SaveType.CHECKPOINT)
    {
        obj._maybe_initialize_trackable();
        // Note: in python the return type of `Trackable._trackable_children` is not fixed.
        // Therefore it uses `convert_to_trackable` to have an extra process.
        return obj._trackable_children(save_type);
    }
    
    public Trackable Root
    {
        get
        {
            if (_root_ref.TryGetTarget(out Trackable res))
            {
                return res;
            }
            else
            {
                throw new InvalidDataException(
                    "Cannot get the object from the weak reference. Please consider if a null reference is passed to the constructor.");
            }
        }
    }
    
    /// <summary>
    /// Returns a list of all nodes and its paths from self.root using a breadth first traversal.
    /// Corresponding to tensorflow/python/checkpoint/trackable_view.Trackable._descendants_with_paths
    /// </summary>
    protected (List<Trackable>, Dictionary<Trackable, IEnumerable<TrackableReference>>) _descendants_with_paths()
    {
        List<Trackable> bfs_sorted = new();
        Queue<Trackable> to_visit = new();
        Dictionary<Trackable, IEnumerable<TrackableReference>> node_paths = new();
        node_paths[this.Root] = new List<TrackableReference>();
        while (!to_visit.empty())
        {
            var current_trackable = to_visit.Dequeue();
            bfs_sorted.Add(current_trackable);
            var children_dict = this.children(current_trackable);
            foreach (var name in children_dict.Keys)
            {
                var dependency = children_dict[name];
                if (!node_paths.ContainsKey(dependency))
                {
                    var list = new List<TrackableReference>(node_paths[current_trackable]);
                    list.Add(new TrackableReference(name, dependency));
                    node_paths[dependency] = list;
                    to_visit.Enqueue(dependency);
                }
            }
        }

        return (bfs_sorted, node_paths);
    }
}