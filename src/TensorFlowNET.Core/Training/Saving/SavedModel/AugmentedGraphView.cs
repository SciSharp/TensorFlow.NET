using System;
using Tensorflow.Checkpoint;
using Tensorflow.Train;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Functions;

namespace Tensorflow;

public class AugmentedGraphView: ObjectGraphView
{
    // private object _children_cache;
    // private object _serialization_cache;
    private List<string> _untraces_functions;
    public AugmentedGraphView(Trackable root): base(root)
    {
        _untraces_functions = new();
    }

    public void set_signature(object signature_map, object wrapped_functions)
    {
        // TODO: cache
        list_children(Root);
    }
    
    public override List<TrackableReference> list_children(Trackable obj, SaveType save_type = SaveType.CHECKPOINT)
    {
        Dictionary<string, Trackable> children = new();
        foreach (var pair in base.list_children(obj, save_type))
        {
            var name = pair.Name;
            var child = pair.Refer;
            children[name] = child;
        }

        if (obj is Function && children.Count == 0)
        {
            _untraces_functions.Add(((Function)obj).Name);
        }

        return children.Select(x => new TrackableReference(x.Key, x.Value)).ToList();
    }

    public override (List<Trackable>, Dictionary<Trackable, IEnumerable<TrackableReference>>) breadth_first_traversal()
    {
        // TODO: implement it if needed.
        return base.breadth_first_traversal();
    }

    public List<(string, Trackable)> list_dependencies(Trackable obj)
    {
        // TODO: deal with cache.
        return obj.deserialization_dependencies(null).Select(x => (x.Key, x.Value)).ToList();
    }

    public Trackable get_child(Trackable obj, string name)
    {
        throw new NotImplementedException();
    }
}