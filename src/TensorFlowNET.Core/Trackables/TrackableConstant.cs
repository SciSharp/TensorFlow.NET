using Google.Protobuf.Collections;
using Tensorflow.Train;
using static Tensorflow.Binding;

namespace Tensorflow.Trackables;

public class TrackableConstant : Trackable
{
    Tensor _constant;
    public TrackableConstant(Tensor constant)
    {
        _constant = constant;
    }

    public static (Tensor, Action<object, object, object>) deserialize_from_proto(SavedObject object_proto,
        Dictionary<string, MapField<string, AttrValue>> operation_attributes)
    {
        var tensor_proto = operation_attributes[object_proto.Constant.Operation]["value"].Tensor;
        var ndarray = tensor_util.MakeNdarray(tensor_proto);
        Tensor imported_constant;
        if (tensor_proto.Dtype == DataType.DtString)
        {
            imported_constant = tf_with(ops.device("CPU"), _ =>
            {
                return constant_op.constant(ndarray);
            });
        }
        else
        {
            imported_constant = constant_op.constant(ndarray);
        }
        return (imported_constant, null);
    }
}
