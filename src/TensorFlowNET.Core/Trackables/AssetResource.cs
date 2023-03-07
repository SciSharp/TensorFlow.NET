using Google.Protobuf.Collections;
using System.IO;
using Tensorflow.Train;

namespace Tensorflow.Trackables;

public class AssetResource : Trackable
{
    public static (Trackable, Action<object, object, object>) deserialize_from_proto(SavedObject object_proto,
        string export_dir,
        RepeatedField<AssetFileDef> asset_file_def,
        Dictionary<string, MapField<string, AttrValue>> operation_attributes)
    {
        var proto = object_proto.Asset;
        var filename = Path.Combine(export_dir, asset_file_def[proto.AssetFileDefIndex].Filename);
        return (new AssetResource(), null);
    }
}
