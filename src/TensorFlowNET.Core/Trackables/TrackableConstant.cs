using Google.Protobuf.Collections;
using Tensorflow.Train;

namespace Tensorflow.Trackables;

public class TrackableConstant : Trackable
{
    Tensor _constant;
    public TrackableConstant(Tensor constant)
    {
        _constant = constant;
    }

    public static (Trackable, Action<object, object, object>) deserialize_from_proto(SavedObject object_proto,
        Dictionary<string, MapField<string, AttrValue>> operation_attributes)
    {
        var tensor_proto = operation_attributes[object_proto.Constant.Operation]["value"].Tensor;
        var ndarray = tensor_util.MakeNdarray(tensor_proto);
        var imported_constant = constant_op.constant(ndarray);
        return (new TrackableConstant(imported_constant), null);
    }
}
