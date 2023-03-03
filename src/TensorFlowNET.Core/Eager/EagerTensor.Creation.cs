using Tensorflow.NumPy;
using System;
using System.Linq;
using static Tensorflow.Binding;

namespace Tensorflow.Eager
{
    public partial class EagerTensor
    {
        public EagerTensor(SafeEagerTensorHandle handle)
        {
            _id = ops.uid();
            _eagerTensorHandle = handle;
        }

        #region scalar eager tensor
        public EagerTensor(bool value) : base(value)
            => NewEagerTensorHandle(_handle);
        public EagerTensor(byte value) : base(value)
            => NewEagerTensorHandle(_handle);
        public EagerTensor(sbyte value) : base(value)
            => NewEagerTensorHandle(_handle);
        public EagerTensor(short value) : base(value)
            => NewEagerTensorHandle(_handle);
        public EagerTensor(int value) : base(value)
            => NewEagerTensorHandle(_handle);
        public EagerTensor(uint value) : base(value)
            => NewEagerTensorHandle(_handle);
        public EagerTensor(long value) : base(value)
            => NewEagerTensorHandle(_handle);
        public EagerTensor(ulong value) : base(value)
            => NewEagerTensorHandle(_handle);
        public EagerTensor(float value) : base(value)
            => NewEagerTensorHandle(_handle);
        public EagerTensor(double value) : base(value)
            => NewEagerTensorHandle(_handle); 
        #endregion

        public EagerTensor(object value, Shape? shape = null, string device_name = null, TF_DataType dtype = TF_DataType.TF_UINT8) : base((float[])value)
            => NewEagerTensorHandle(_handle);

        public EagerTensor(Shape shape, TF_DataType dtype) : base(shape, dtype)
            => NewEagerTensorHandle(_handle);

        public EagerTensor(Array array, Shape shape) : base(array, shape)
            => NewEagerTensorHandle(_handle);

        public EagerTensor(byte[] bytes, Shape shape, TF_DataType dtype) : base(bytes, shape, dtype)
            => NewEagerTensorHandle(_handle);

        public EagerTensor(IntPtr data_ptr, Shape shape, TF_DataType dtype) : base(data_ptr, shape, dtype)
            => NewEagerTensorHandle(_handle);

        void NewEagerTensorHandle(SafeTensorHandle h)
        {
            _id = ops.uid();
            _eagerTensorHandle = c_api.TFE_NewTensorHandle(h, tf.Status);
#if TRACK_TENSOR_LIFE
            Console.WriteLine($"New EagerTensor {_eagerTensorHandle}");
#endif
            tf.Status.Check(true);
        }

        public void Resolve()
        {
            if (_handle != null)
                return;
            _handle = c_api.TFE_TensorHandleResolve(_eagerTensorHandle, tf.Status);
            tf.Status.Check(true);
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
    }
}
