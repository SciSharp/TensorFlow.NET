using NumSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow.Eager
{
    public partial class EagerTensor : Tensor
    {
        public IntPtr EagerTensorHandle;
        public override string Device => c_api.StringPiece(c_api.TFE_TensorHandleDeviceName(EagerTensorHandle, tf.status.Handle));

        public override int rank => c_api.TFE_TensorHandleNumDims(EagerTensorHandle, tf.status.Handle);

        public static int GetRank(IntPtr handle)
        {
            var tfe_tensor_handle = c_api.TFE_EagerTensorHandle(handle);
            return c_api.TFE_TensorHandleNumDims(tfe_tensor_handle, tf.status.Handle);
        }

        public static int[] GetDims(IntPtr handle)
        {
            var tfe_tensor_handle = c_api.TFE_EagerTensorHandle(handle);
            var dims = new int[c_api.TFE_TensorHandleNumDims(tfe_tensor_handle, tf.status.Handle)];
            for (int i = 0; i < dims.Length; i++)
                dims[i] = c_api.TFE_TensorHandleDim(tfe_tensor_handle, i, tf.status.Handle);
            return dims;
        }

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
                    return string.Join(string.Empty, nd.ToArray<byte>()
                        .Select(x => x < 32 || x > 127 ? "\\x" + x.ToString("x") : Convert.ToChar(x).ToString()));
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
