using System.Runtime.CompilerServices;
using Tensorflow.Train;

namespace Tensorflow.Trackables;

public class RestoredResource : TrackableResource
{
    public static (Trackable, Action<object, object, object>) deserialize_from_proto()
    {
        return (null, null);
    }
}
