using Google.Protobuf.Collections;
using Tensorflow.Train;

namespace Tensorflow.Trackables;

public class RestoredResource : TrackableResource
{
    public static (Trackable, Action<object, object, object>) deserialize_from_proto(SavedObject object_proto,
        Dictionary<string, MapField<string, AttrValue>> operation_attributes)
    {
        return (new RestoredResource(), null);
    }
}
