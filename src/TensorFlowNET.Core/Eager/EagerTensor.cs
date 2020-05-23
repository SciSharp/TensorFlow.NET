using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow.Eager
{
    public partial class EagerTensor : Tensor
    {
        Status status = new Status();
        IntPtr tfe_tensor_handle;
        public IntPtr EagerTensorHandle { get; set; }
        public override string Device => c_api.StringPiece(c_api.TFE_TensorHandleDeviceName(tfe_tensor_handle, status));

        public override string ToString()
        {
            switch (rank)
            {
                case -1:
                    return $"tf.Tensor: shape=<unknown>, dtype={dtype.as_numpy_name()}, numpy={GetFormattedString(dtype, numpy())}";
                case 0:
                    return $"tf.Tensor: shape=(), dtype={dtype.as_numpy_name()}, numpy={GetFormattedString(dtype, numpy())}";
                default:
                    return $"tf.Tensor: shape=({string.Join(",", shape)}), dtype={dtype.as_numpy_name()}, numpy={GetFormattedString(dtype, numpy())}";
            }
        }

        public static string GetFormattedString(TF_DataType dtype, NDArray nd)
        {
            if (nd.size == 0)
                return "[]";

            switch (dtype)
            {
                case TF_DataType.TF_STRING:
                    return $"b'{(string)nd}'";
                case TF_DataType.TF_BOOL:
                    return (nd.GetByte(0) > 0).ToString();
                case TF_DataType.TF_RESOURCE:
                    return "<unprintable>";
                default:
                    return nd.ToString();
            }
        }
    }
}
