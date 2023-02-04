using System;
using System.Collections.Generic;
using System.Linq;
using Serilog.Debugging;
using Tensorflow.Keras.Saving.SavedModel;
using Tensorflow.Train;

namespace Tensorflow.Checkpoint;

public class ObjectGraphView: TrackableView, ICloneable
{
    protected IEnumerable<TrackableReference>? _attached_dependencies;
    // TODO: attached_dependencies
    public ObjectGraphView(Trackable root, IEnumerable<TrackableReference>? attached_dependencies = null): base(root)
    {
        _attached_dependencies = attached_dependencies;
    }

    public object Clone()
    {
        // TODO: Implement real deep copy corresponding to tensorflow/python/checkpoint/graph_view.ObjectGraphView.__deepcopy__
        return new ObjectGraphView(Root, _attached_dependencies);
    }

    public virtual List<TrackableReference> list_children(Trackable obj, SaveType save_type = SaveType.CHECKPOINT, IDictionary<string, IDictionary<Trackable, ISerializedAttributes>>? serialization_cache = null)
    {
        List<TrackableReference> res = base.children(obj, save_type, serialization_cache)
            .Select(x => new TrackableReference(x.Key, x.Value)).ToList();
        // Check the reference, not value.
        if (obj == Root && _attached_dependencies is not null)
        {
            res.AddRange(_attached_dependencies);
        }

        return res;
    }
    
    public override IDictionary<string, Trackable> children(Trackable obj, SaveType save_type = SaveType.CHECKPOINT, IDictionary<string, IDictionary<Trackable, ISerializedAttributes>>? serialization_cache = null)
    {
        return list_children(obj, save_type, serialization_cache).ToDictionary(x => x.Name, x => x.Refer);
    }
    
    public IEnumerable<TrackableReference>? AttachedDependencies
    {
        get => _attached_dependencies;
    }

    public virtual (IList<Trackable>, IDictionary<Trackable, IEnumerable<TrackableReference>>) breadth_first_traversal()
    {
        return base._descendants_with_paths();
    }

    // TODO: complete the implementation
    public void serialize_object_graph(object? saveables_cache = null)
    {
        throw new NotImplementedException();
    }
    
    // TODO: complete the implementation
    public void frozen_saveable_objects(object? object_map = null, object? to_graph = null, object call_with_mapped_captures = null)
    {
        throw new NotImplementedException();
    }
}
