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

        public EagerTensor(IntPtr handle) : base(IntPtr.Zero)
        {
            EagerTensorHandle = handle;
            Resolve();
        }

        public EagerTensor(string value, string device_name) : base(value)
        {
            EagerTensorHandle = c_api.TFE_NewTensorHandle(_handle, tf.status);
            Resolve();
        }
        
        public EagerTensor(NDArray value, string device_name) : base(value)
        {
            EagerTensorHandle = c_api.TFE_NewTensorHandle(_handle, tf.status);
            Resolve();
        }

        public EagerTensor Resolve()
        {
            _id = get_uid();

            if (_handle == IntPtr.Zero)
                _handle = c_api.TFE_TensorHandleResolve(EagerTensorHandle, tf.status);

            //print($"new Tensor {Id} {_handle.ToString("x16")}");
            //print($"new TensorHandle {Id} {EagerTensorHandle.ToString("x16")}");

            return this;
        }

        public override IntPtr ToPointer()
            => EagerTensorHandle;

        protected override void DisposeUnmanagedResources(IntPtr handle)
        {
            //print($"deleting DeleteTensorHandle {Id} {_handle.ToString("x16")}");
            c_api.TF_DeleteTensor(_handle);
            //print($"deleting DeleteTensorHandle {Id} {EagerTensorHandle.ToString("x16")}");
            c_api.TFE_DeleteTensorHandle(EagerTensorHandle);
        }

        static long _uid = 0;
        long get_uid() => _uid++;
    }
}
