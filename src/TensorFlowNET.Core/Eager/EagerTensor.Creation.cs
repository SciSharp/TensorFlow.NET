using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using static Tensorflow.Binding;

namespace Tensorflow.Eager
{
    public partial class EagerTensor : Tensor
    {
        public EagerTensor() : base(IntPtr.Zero)
        {
            
        }

        public EagerTensor(SafeTensorHandleHandle handle) : base(IntPtr.Zero)
        {
            EagerTensorHandle = handle;
            Resolve();
        }

        public EagerTensor(string value, string device_name) : base(value)
        {
            EagerTensorHandle = c_api.TFE_NewTensorHandle(_handle, tf.status.Handle);
            Resolve();
        }

        public EagerTensor(byte[] value, string device_name, TF_DataType dtype) : base(value, dType: dtype)
        {
            EagerTensorHandle = c_api.TFE_NewTensorHandle(_handle, tf.status.Handle);
            Resolve();
        }

        public EagerTensor(string[] value, string device_name) : base(value)
        {
            EagerTensorHandle = c_api.TFE_NewTensorHandle(_handle, tf.status.Handle);
            Resolve();
        }

        public EagerTensor(NDArray value, string device_name) : base(value)
        {
            EagerTensorHandle = c_api.TFE_NewTensorHandle(_handle, tf.status.Handle);
            Resolve();
        }

        public EagerTensor Resolve()
        {
            _id = ops.uid();

            if (_handle == IntPtr.Zero)
                _handle = c_api.TFE_TensorHandleResolve(EagerTensorHandle, tf.status.Handle);

            //print($"new Tensor {Id} {_handle.ToString("x16")}");
            //print($"new TensorHandle {Id} {EagerTensorHandle.ToString("x16")}");

            return this;
        }

        public override IntPtr ToPointer()
            => EagerTensorHandle?.DangerousGetHandle() ?? IntPtr.Zero;

        protected override void DisposeManagedResources()
        {
            base.DisposeManagedResources();

            //print($"deleting DeleteTensorHandle {Id} {EagerTensorHandle.ToString("x16")}");
            EagerTensorHandle.Dispose();
        }

        protected override void DisposeUnmanagedResources(IntPtr handle)
        {
            //print($"deleting DeleteTensorHandle {Id} {_handle.ToString("x16")}");
            c_api.TF_DeleteTensor(_handle);
        }
    }
}
