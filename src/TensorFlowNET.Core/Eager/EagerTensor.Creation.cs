using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow.Eager
{
    public partial class EagerTensor : Tensor
    {
        public EagerTensor(IntPtr handle) : base(handle)
        {
            EagerTensorHandle = handle;
            tfe_tensor_handle = c_api.EagerTensor_Handle(handle);
            _handle = c_api.TFE_TensorHandleResolve(tfe_tensor_handle, status);
        }

        /*public EagerTensor(int value, string device_name) : base(value)
        {
            tfe_tensor_handle = c_api.TFE_NewTensorHandle(_handle, status);
            EagerTensorHandle = c_api.TFE_EagerTensorFromHandle(tf.context, tfe_tensor_handle);
        }

        public EagerTensor(long value, string device_name) : base(value)
        {
            tfe_tensor_handle = c_api.TFE_NewTensorHandle(_handle, status);
            EagerTensorHandle = c_api.TFE_EagerTensorFromHandle(tf.context, tfe_tensor_handle);
        }

        public EagerTensor(float value, string device_name) : base(value)
        {
            tfe_tensor_handle = c_api.TFE_NewTensorHandle(_handle, status);
            EagerTensorHandle = c_api.TFE_EagerTensorFromHandle(tf.context, tfe_tensor_handle);
        }*/
        
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
    }
}
