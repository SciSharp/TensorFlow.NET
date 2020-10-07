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
            EagerTensorHandle = c_api.TFE_NewTensorHandle(_handle, tf.Status.Handle);
            Resolve();
        }

        public EagerTensor(byte[] value, string device_name, TF_DataType dtype) : base(value, dType: dtype)
        {
            EagerTensorHandle = c_api.TFE_NewTensorHandle(_handle, tf.Status.Handle);
            Resolve();
        }

        public EagerTensor(string[] value, string device_name) : base(value)
        {
            EagerTensorHandle = c_api.TFE_NewTensorHandle(_handle, tf.Status.Handle);
            Resolve();
        }

        public EagerTensor(NDArray value, string device_name) : base(value)
        {
            EagerTensorHandle = c_api.TFE_NewTensorHandle(_handle, tf.Status.Handle);
            Resolve();
        }

        public EagerTensor Resolve()
        {
            _id = ops.uid();

            if (_handle == IntPtr.Zero)
                _handle = c_api.TFE_TensorHandleResolve(EagerTensorHandle, tf.Status.Handle);

            //print($"new Tensor {Id} {_handle.ToString("x16")}");
            //print($"new TensorHandle {Id} {EagerTensorHandle.ToString("x16")}");

            return this;
        }

        /// <summary>
        /// _create_substitute_placeholder
        /// </summary>
        /// <returns></returns>
        public Tensor AsPlaceholder(string name = null)
        {
            Tensor placeholder = null;
            tf_with(ops.control_dependencies(null), delegate
            {
                placeholder = tf.placeholder(dtype, shape: shape, name: name ?? this.name);
            });
            // custom_gradient.copy_handle_data(value, placeholder)
            return placeholder;
        }

        void copy_handle_data()
        {

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
            base.DisposeUnmanagedResources(handle);
            //print($"deleting DeleteTensorHandle {Id} {_handle.ToString("x16")}");
        }
    }
}
