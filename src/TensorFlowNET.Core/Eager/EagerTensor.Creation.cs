using NumSharp;
using System;
using static Tensorflow.Binding;

namespace Tensorflow.Eager
{
    public partial class EagerTensor
    {
        SafeOpHandle _opHandle;

        public EagerTensor(SafeTensorHandleHandle handle, SafeOpHandle opHandle) : base(IntPtr.Zero)
        {
            _opHandle = opHandle;
            EagerTensorHandle = handle;
            Resolve();
        }

        public EagerTensor(string value, string device_name) : base(value)
        {
            SetEagerTensorHandleAndResolve();
        }

        public EagerTensor(byte[] value, string device_name, TF_DataType dtype) : base(value, dType: dtype)
        {
            SetEagerTensorHandleAndResolve();
        }

        public EagerTensor(string[] value, string device_name) : base(value)
        {
            SetEagerTensorHandleAndResolve();
        }

        public EagerTensor(NDArray value, string device_name) : base(value)
        {
            SetEagerTensorHandleAndResolve();
        }

        void SetEagerTensorHandleAndResolve()
        {
            EagerTensorHandle = c_api.TFE_NewTensorHandle(_handle, tf.Status.Handle);
            Resolve();
        }

        public EagerTensor Resolve()
        {
            _id = ops.uid();

            if (_handle == IntPtr.Zero)
                _handle = c_api.TFE_TensorHandleResolve(EagerTensorHandle, tf.Status.Handle);
#if TRACK_TENSOR_LIFE
            print($"New TensorHandle {Id} 0x{_handle.ToString("x16")}");
            print($"New EagerTensorHandle {Id} {EagerTensorHandle}");
#endif
            return this;
        }

        /// <summary>
        /// _create_substitute_placeholder
        /// </summary>
        /// <returns></returns>
        public Tensor AsPlaceholder(string name = null)
        {
            var placeholder = tf_with(ops.control_dependencies(null), _ => tf.placeholder(dtype, name: name));
            copy_handle_data(placeholder);
            return placeholder;
        }

        public Tensor AsConstant(string name = null)
        {
            return tf_with(ops.control_dependencies(null), _ => tf.constant(numpy(), name: name));
        }

        void copy_handle_data(Tensor target_t)
        {
            if (target_t.dtype == TF_DataType.TF_RESOURCE ||
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
        }

        protected override void DisposeUnmanagedResources(IntPtr handle)
        {
            base.DisposeUnmanagedResources(handle);
            
            EagerTensorHandle.Dispose();

            if (_opHandle != null)
                _opHandle.Dispose();
        }
    }
}
