using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Gradients;

namespace Tensorflow.Eager
{
    public partial class EagerTensor : Tensor
    {
        public EagerTensor() : base(IntPtr.Zero)
        {
            EagerTensorHandle = c_api.TFE_NewEagerTensor();
            // _id = c_api.TFE_EagerTensorId(EagerTensorHandle);
            // print($"new EagerTensorHandle {EagerTensorHandle.ToString("x16")} {Id}");
        }

        public EagerTensor(IntPtr handle) : base(IntPtr.Zero)
        {
            EagerTensorHandle = handle;
            Resolve();
            // print($"new EagerTensorHandle {EagerTensorHandle.ToString("x16")} {Id}");
        }

        public EagerTensor(string value, string device_name) : base(value)
        {
            EagerTensorHandle = c_api.TFE_NewEagerTensor();
            tfe_tensor_handle = c_api.TFE_NewTensorHandle(_handle, status);
            c_api.TFE_SetEagerTensorHandle(EagerTensorHandle, tfe_tensor_handle);
            Resolve();
            // print($"new EagerTensorHandle {EagerTensorHandle.ToString("x16")} {Id}");
        }
        
        public EagerTensor(NDArray value, string device_name) : base(value)
        {
            EagerTensorHandle = c_api.TFE_NewEagerTensor();
            tfe_tensor_handle = c_api.TFE_NewTensorHandle(_handle, status);
            c_api.TFE_SetEagerTensorHandle(EagerTensorHandle, tfe_tensor_handle);
            Resolve();
            // print($"new EagerTensorHandle {EagerTensorHandle.ToString("x16")} {Id}");
        }

        public EagerTensor Resolve()
        {
            if (tfe_tensor_handle == IntPtr.Zero)
                tfe_tensor_handle = c_api.TFE_EagerTensorHandle(EagerTensorHandle);

            if (_handle == IntPtr.Zero)
                _handle = c_api.TFE_TensorHandleResolve(tfe_tensor_handle, status);

            _id = c_api.TFE_EagerTensorId(EagerTensorHandle);

            GarbageCollector.Increase(_handle, GCItemType.TensorHandle);
            GarbageCollector.Increase(tfe_tensor_handle, GCItemType.LocalTensorHandle);
            GarbageCollector.Increase(EagerTensorHandle, GCItemType.EagerTensorHandle);

            return this;
        }

        protected override void DisposeUnmanagedResources(IntPtr handle)
        {
            GarbageCollector.Decrease(_handle);
            GarbageCollector.Decrease(tfe_tensor_handle);
            GarbageCollector.Decrease(EagerTensorHandle);
        }
    }
}
