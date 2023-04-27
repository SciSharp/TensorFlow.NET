using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Eager;
using static Tensorflow.Binding;

namespace Tensorflow.NumPy
{
    public partial class NDArray
    {
        protected NDArray() { }
        public NDArray(bool value) : base(value) => NewEagerTensorHandle(); 
        public NDArray(byte value) : base(value) => NewEagerTensorHandle();
        public NDArray(short value) : base(value) => NewEagerTensorHandle();
        public NDArray(int value) : base(value) => NewEagerTensorHandle();
        public NDArray(long value) : base(value) => NewEagerTensorHandle();
        public NDArray(float value) : base(value) => NewEagerTensorHandle();
        public NDArray(double value) : base(value) => NewEagerTensorHandle(); 

        public NDArray(Array value, Shape? shape = null) : base(value, shape) 
            => NewEagerTensorHandle();

        public NDArray(Shape shape, TF_DataType dtype = TF_DataType.TF_DOUBLE) : base(shape, dtype: dtype) 
            => NewEagerTensorHandle();

        public NDArray(byte[] bytes, Shape shape, TF_DataType dtype) : base(bytes, shape, dtype) 
            => NewEagerTensorHandle();

        public NDArray(int[] value, Shape? shape = null) : base(value, shape)
            => NewEagerTensorHandle();

        public NDArray(long[] value, Shape? shape = null) : base(value, shape) 
            => NewEagerTensorHandle();

        public NDArray(IntPtr address, Shape shape, TF_DataType dtype) : base(address, shape, dtype) 
            => NewEagerTensorHandle();

        public NDArray(Tensor tensor, bool clone = false) : base(tensor.Handle, clone: clone) 
        {
            if (_handle is null)
            {
                tensor = tf.get_default_session().eval(tensor);
                _handle = tensor.Handle;
            }

            NewEagerTensorHandle();
        }

        public static NDArray Scalar<T>(T value) where T : unmanaged
            => value switch
            {
                bool val => new NDArray(val),
                byte val => new NDArray(val),
                int val => new NDArray(val),
                long val => new NDArray(val),
                float val => new NDArray(val),
                double val => new NDArray(val),
                _ => throw new NotImplementedException("")
            };

        /// <summary>
        /// Reuse the existing memory instead of copying it.
        /// </summary>
        /// <param name="data_ptr"></param>
        /// <param name="shape"></param>
        /// <param name="dtype"></param>
        /// <param name="deallocator"></param>
        protected void InitWithExistingMemory(IntPtr data_ptr, Shape shape, TF_DataType dtype, c_api.DeallocatorV2 deallocator)
        {
            _handle = c_api.TF_NewTensor(TF_DataType.TF_STRING, shape.dims, shape.ndim, data_ptr, (ulong)(shape.size * dtype.get_datatype_size()), deallocator, IntPtr.Zero);
            tensor_util.DangerousManuallySetTensorDType(_handle, dtype);
            NewEagerTensorHandle();
        }

        void NewEagerTensorHandle()
        {
            if (_handle is not null)
            {
                _eagerTensorHandle = c_api.TFE_NewTensorHandle(_handle, tf.Status);
            }
        }
    }
}
