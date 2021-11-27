using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Eager;
using static Tensorflow.Binding;

namespace Tensorflow.NumPy
{
    public partial class NDArray
    {
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

        public NDArray(long[] value, Shape? shape = null) : base(value, shape) 
            => NewEagerTensorHandle();

        public NDArray(IntPtr address, Shape shape, TF_DataType dtype) : base(address, shape, dtype) 
            => NewEagerTensorHandle();

        public NDArray(Tensor tensor, bool clone = false) : base(tensor.Handle, clone: clone) 
        {
            if (_handle is null)
            {
                tensor = tf.defaultSession.eval(tensor);
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

        void NewEagerTensorHandle()
        {
            if (_handle is not null)
            {
                _eagerTensorHandle = c_api.TFE_NewTensorHandle(_handle, tf.Status.Handle);
            }
        }
    }
}
