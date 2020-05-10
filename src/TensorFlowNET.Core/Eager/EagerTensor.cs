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
        TFE_TensorHandle tfe_tensor_handle;
        public IntPtr EagerTensorHandle { get; set; }

        public EagerTensor(IntPtr handle) : base(handle)
        {
            EagerTensorHandle = handle;
            tfe_tensor_handle = c_api.EagerTensor_Handle(handle);
            _handle = c_api.TFE_TensorHandleResolve(tfe_tensor_handle, status);
        }

        public EagerTensor(TFE_TensorHandle handle) : base(handle)
        {
            tfe_tensor_handle = handle;
            _handle = c_api.TFE_TensorHandleResolve(tfe_tensor_handle, status);
            EagerTensorHandle = c_api.TFE_EagerTensorFromHandle(tf.context, tfe_tensor_handle);
        }

        public EagerTensor(string value, string device_name) : base(value)
        {
            tfe_tensor_handle = c_api.TFE_NewTensorHandle(_handle, status);
            EagerTensorHandle = c_api.TFE_EagerTensorFromHandle(tf.context, tfe_tensor_handle);
        }

        public EagerTensor(NDArray value, string device_name) : base(value)
        {
            tfe_tensor_handle = c_api.TFE_NewTensorHandle(_handle, status);
            EagerTensorHandle = c_api.TFE_EagerTensorFromHandle(tf.context, tfe_tensor_handle);
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
