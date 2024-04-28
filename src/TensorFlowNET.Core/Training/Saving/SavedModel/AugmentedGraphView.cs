using System;
using Tensorflow.Checkpoint;
using Tensorflow.Train;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Functions;
using Tensorflow.Keras.Saving.SavedModel;

namespace Tensorflow;

public class AugmentedGraphView: ObjectGraphView
{
    private Dictionary<Trackable, IDictionary<string, Trackable>> _children_cache;
    private Dictionary<string, IDictionary<Trackable, ISerializedAttributes>> _serialization_cache;
    private List<string> _untraces_functions;
    private Dictionary<ConcreteFunction, ConcreteFunction> _wrapped_functions;
    public AugmentedGraphView(Trackable root): base(root)
    {
        _children_cache= new Dictionary<Trackable, IDictionary<string, Trackable>>();
        _serialization_cache = new Dictionary<string, IDictionary<Trackable, ISerializedAttributes>>();
        _untraces_functions = new List<string>();
        _wrapped_functions = new Dictionary<ConcreteFunction, ConcreteFunction>();
    }

    public void set_signature(SignatureMap signature_map, IDictionary<ConcreteFunction, ConcreteFunction> wrapped_functions)
    {
        list_children(Root);
        var name = SignatureSerializationUtils.SIGNATURE_ATTRIBUTE_NAME;
        if (!_children_cache.ContainsKey(Root))
        {
            _children_cache[Root] = new Dictionary<string, Trackable>();
        }
        _children_cache[Root][name] = signature_map;
        _wrapped_functions =  _wrapped_functions.Concat(wrapped_functions).ToDictionary(x => x.Key, x => x.Value);
    }
    
    public override List<TrackableReference> list_children(Trackable obj, SaveType save_type = SaveType.SAVEDMODEL, IDictionary<string, IDictionary<Trackable, ISerializedAttributes>>? serialization_cache = null)
    {
        if(serialization_cache is not null)
        {
            throw new ValueError("Serialization cache should not be passed to `AugmentedGraphView.list_children`, please either remove the parameter or use `ObjectGraphView.list_children`.");
        }

        if (!_children_cache.ContainsKey(obj))
        {
            Dictionary<string, Trackable> children = new Dictionary<string, Trackable>();
            _children_cache[obj] = children;
            foreach (var pair in base.list_children(obj, SaveType.SAVEDMODEL, _serialization_cache))
            {
                var name = pair.Name;
                var child = pair.Refer;
                if(child is ConcreteFunction)
                {
                    child = maybe_uncache_variable_captures((ConcreteFunction)child);
                }
                children[name] = child;
            }

            if (obj is Function && children.Count == 0)
            {
                _untraces_functions.Add(((Function)obj).Name);
            }
        }

        List<TrackableReference> res = new();
        foreach(var pair in _children_cache[obj])
        {
            res.Add(new TrackableReference(pair.Key, pair.Value));
        }

        return res;
    }

    private ConcreteFunction maybe_uncache_variable_captures(ConcreteFunction concrete_function)
    {
        if (_wrapped_functions.ContainsKey(concrete_function))
        {
            return _wrapped_functions[concrete_function];
        }
        // skip the process here because of lack of feature.
        // In the future, we may add an attribute which could specify if the variable is supposed to be cached.
        //foreach(var capture in concrete_function.CapturedInputs)
        //{

        //}
        return concrete_function;
    }

    public override (IList<Trackable>, IDictionary<Trackable, IEnumerable<TrackableReference>>) breadth_first_traversal()
    {
        void merged_trackable(Trackable x)
        {
            // TODO: complete it with new definitions `Asset` and `TrackableConstant`.
        }

        var trackable_objects = base.breadth_first_traversal();

        foreach(var obj in _children_cache.Keys)
        {
            // skip the deletion of cache (maybe do it later).
            foreach(var pair in _children_cache[obj])
            {
                merged_trackable(pair.Value);
            }
        }

        return base.breadth_first_traversal();
    }

    public List<(string, Trackable)> list_dependencies(Trackable obj)
    {
        if (!_children_cache.TryGetValue(obj, out var children))
        {
            children= new Dictionary<string, Trackable>();
        }

        List<(string, Trackable)> res = new();
        foreach(var pair in obj.deserialization_dependencies(children))
        {
            res.Add((pair.Key, pair.Value));
        }
        return res;
    }

    public Trackable get_child(Trackable obj, string name)
    {
        return _children_cache[obj][name];
    }
}
