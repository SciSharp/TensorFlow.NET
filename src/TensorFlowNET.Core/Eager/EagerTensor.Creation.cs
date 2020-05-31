using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow.Eager
{
    public partial class EagerTensor : Tensor
    {
        public EagerTensor() : base(IntPtr.Zero)
        {
            EagerTensorHandle = c_api.TFE_NewEagerTensor();
        }

        public EagerTensor(IntPtr handle) : base(IntPtr.Zero)
        {
            EagerTensorHandle = handle;
            Resolve();
        }

        public EagerTensor(string value, string device_name) : base(value)
        {
            EagerTensorHandle = c_api.TFE_NewEagerTensor();
            tfe_tensor_handle = c_api.TFE_NewTensorHandle(_handle, status);
            c_api.TFE_SetEagerTensorHandle(EagerTensorHandle, tfe_tensor_handle);
            Resolve();
        }
        
        public EagerTensor(NDArray value, string device_name) : base(value)
        {
            EagerTensorHandle = c_api.TFE_NewEagerTensor();
            tfe_tensor_handle = c_api.TFE_NewTensorHandle(_handle, status);
            c_api.TFE_SetEagerTensorHandle(EagerTensorHandle, tfe_tensor_handle);
            Resolve();
        }

        public EagerTensor Resolve()
        {
            _id = c_api.TFE_EagerTensorId(EagerTensorHandle);

            if (tfe_tensor_handle == IntPtr.Zero)
                tfe_tensor_handle = c_api.TFE_EagerTensorHandle(EagerTensorHandle);

            if (_handle == IntPtr.Zero)
                _handle = c_api.TFE_TensorHandleResolve(tfe_tensor_handle, status);

            /*print($"new Tensor {Id} {_handle.ToString("x16")}");
            print($"new TensorHandle {Id} {tfe_tensor_handle.ToString("x16")}");
            print($"new EagerTensor {Id} {EagerTensorHandle.ToString("x16")}");*/

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

            /*print($"deleting DeleteTensorHandle {Id} {_handle.ToString("x16")}");
            c_api.TF_DeleteTensor(_handle);
            print($"deleting DeleteTensorHandle {Id} {tfe_tensor_handle.ToString("x16")}");
            c_api.TFE_DeleteTensorHandle(tfe_tensor_handle);
            print($"deleting DeleteEagerTensor {Id} {EagerTensorHandle.ToString("x16")}");
            c_api.TFE_DeleteEagerTensor(EagerTensorHandle);*/
        }
    }
}
