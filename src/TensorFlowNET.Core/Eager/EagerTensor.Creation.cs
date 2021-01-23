using NumSharp;
using System;
using System.Linq;
using static Tensorflow.Binding;

namespace Tensorflow.Eager
{
    public partial class EagerTensor
    {
        public EagerTensor(SafeTensorHandleHandle handle)
        {
            _id = ops.uid();
            EagerTensorHandle = handle;
            Resolve();
        }

        public EagerTensor(string value, string device_name) : base(value)
        {
            NewEagerTensorHandle(_handle);
        }

        public EagerTensor(byte[] value, string device_name, TF_DataType dtype) : base(value, dType: dtype)
        {
            NewEagerTensorHandle(_handle);
        }

        public EagerTensor(string[] value, string device_name) : base(value)
        {
            NewEagerTensorHandle(_handle);
        }

        public EagerTensor(NDArray value, string device_name) : base(value)
        {
            NewEagerTensorHandle(_handle);
        }

        void NewEagerTensorHandle(IntPtr h)
        {
            _id = ops.uid();
            EagerTensorHandle = c_api.TFE_NewTensorHandle(h, tf.Status.Handle);
            tf.Status.Check(true);
#if TRACK_TENSOR_LIFE
            print($"New EagerTensorHandle {EagerTensorHandle} {Id} From 0x{h.ToString("x16")}");
#endif
        }

        private void Resolve()
        {
            _handle = c_api.TFE_TensorHandleResolve(EagerTensorHandle, tf.Status.Handle);
            tf.Status.Check(true);
#if TRACK_TENSOR_LIFE
            print($"Take EagerTensorHandle {EagerTensorHandle} {Id} Resolving 0x{_handle.ToString("x16")}");
#endif
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

        protected override void DisposeUnmanagedResources(IntPtr handle)
        {
            base.DisposeUnmanagedResources(handle);
            EagerTensorHandle.Dispose();
        }
    }
}
