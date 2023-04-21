using System;
using System.Diagnostics;
using Tensorflow.Train;
using Tensorflow.Training;

namespace Tensorflow;

public class RevivedTypes
{
    private static Dictionary<string, ITrackableWrapper> _registered_revived_creator = new();
    static RevivedTypes()
    {
        var list_wrapper = new ListWrapper(new Trackable[] { });
        _registered_revived_creator[list_wrapper.Identifier] = list_wrapper;
        var dict_wrapper = new DictWrapper(new Dictionary<object, Trackable>());
        _registered_revived_creator[dict_wrapper.Identifier] = dict_wrapper;
    }
    /// <summary>
    /// Create a SavedUserObject from a trackable object.
    /// </summary>
    /// <param name="obj"></param>
    /// <returns></returns>
    public static SavedUserObject? serialize(Trackable obj)
    {
        // TODO(Rinne): complete the implementation.
        return null;
    }

    public static (Trackable, Action<object, object, object>) deserialize(SavedUserObject proto)
    {
        if(_registered_revived_creator.TryGetValue(proto.Identifier, out var wrapper))
        {
            return (wrapper.FromProto(proto), (x, y, z) =>
                {
                    if (x is not ITrackableWrapper trackable)
                    {
                        throw new TypeError($"The type is expected to be `ITrackableWrapper`, but got {x.GetType()}.");
                    }
                    Debug.Assert(y is string);
                    trackable.SetValue(y, z);
                }
            );
        }
        else
        {
            return (null, null);
        }
    }

    public static void RegisterRevivedTypeCreator(string identifier, ITrackableWrapper obj)
    {
        _registered_revived_creator[identifier] = obj;
    }
}
