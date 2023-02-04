using System.Collections.Generic;
using System.Linq;
using Tensorflow.Keras.Engine;
using Newtonsoft.Json;
using Tensorflow.Train;

namespace Tensorflow.Keras.Saving.SavedModel;

public abstract class SavedModelSaver
{
    protected Trackable _obj;
    public SavedModelSaver(Trackable obj)
    {
        _obj = obj;
    }

    public abstract string ObjectIdentifier { get; }
    public abstract string TrackingMetadata { get; }

    public abstract IDictionary<string, Trackable> objects_to_serialize(
        IDictionary<string, IDictionary<Trackable, ISerializedAttributes>> serialization_cache);

    public abstract IDictionary<string, Trackable> functions_to_serialize(
        IDictionary<string, IDictionary<Trackable, ISerializedAttributes>> serialization_cache);

    public IDictionary<string, Trackable> trackable_children(IDictionary<string, IDictionary<Trackable, ISerializedAttributes>> serialization_cache)
    {
        if (!KerasSavedModelUtils.ShouldHaveTraces)
        {
            return new Dictionary<string, Trackable>();
        }

        var children = objects_to_serialize(serialization_cache);
        return children.Concat(functions_to_serialize(serialization_cache).ToDictionary(x => x.Key, x => (Trackable)x.Value))
            .ToDictionary(x => x.Key, x => x.Value);
    }
}
