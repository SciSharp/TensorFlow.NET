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
                placeholder = tf.placeholder(dtype, name: name);
            });
            copy_handle_data(placeholder);
            return placeholder;
        }

        public Tensor AsContatnt(string name = null)
        {
            Tensor constant = null;
            tf_with(ops.control_dependencies(null), delegate
            {
                constant = tf.constant(numpy(), name: name);
            });
            return constant;
        }

        void copy_handle_data(Tensor target_t)
        {
            if(target_t.dtype == TF_DataType.TF_RESOURCE ||
                target_t.dtype == TF_DataType.TF_VARIANT)
            {
                // need to export
                // c_api.TF_GraphSetOutputHandleShapesAndTypes(target_t.graph, target_t._as_tf_output(), 0, new IntPtr[0], new int[0], new DataType[0], tf.Status.Handle);
            }
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
