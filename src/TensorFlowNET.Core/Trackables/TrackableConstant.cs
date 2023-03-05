using Tensorflow.Train;

namespace Tensorflow.Trackables;

public class TrackableConstant : Trackable
{
    public static (Trackable, Action<object, object, object>) deserialize_from_proto()
    {
        return (null, null);
    }
}
