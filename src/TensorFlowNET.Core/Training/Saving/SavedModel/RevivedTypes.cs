using System;
using Tensorflow.Train;

namespace Tensorflow;

public class RevivedTypes
{
    /// <summary>
    /// Create a SavedUserObject from a trackable object.
    /// </summary>
    /// <param name="obj"></param>
    /// <returns></returns>
    public static SavedUserObject? serialize(Trackable obj)
    {
        // TODO: complete the implementation.
        return null;
    }

    public static Tuple<Trackable, Action<object, object, object>> deserialize(object proto)
    {
        // TODO: complete the implementation.
        return null;
    }
}
