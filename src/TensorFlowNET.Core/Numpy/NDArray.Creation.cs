using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Eager;
using static Tensorflow.Binding;

namespace Tensorflow.NumPy
{
    public partial class NDArray
    {
        public NDArray(bool value) => Init(value);
        public NDArray(byte value) => Init(value);
        public NDArray(short value) => Init(value);
        public NDArray(int value) => Init(value);
        public NDArray(long value) => Init(value);
        public NDArray(float value) => Init(value);
        public NDArray(double value) => Init(value);
        public NDArray(Array value, Shape? shape = null) => Init(value, shape);
        public NDArray(Shape shape, TF_DataType dtype = TF_DataType.TF_DOUBLE) => Init(shape, dtype: dtype);
        public NDArray(Tensor value, Shape? shape = null) => Init(value, shape);

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

        void Init<T>(T value) where T : unmanaged
        {
            _tensor = value switch
            {
                bool val => new Tensor(val),
                byte val => new Tensor(val),
                int val => new Tensor(val),
                long val => new Tensor(val),
                float val => new Tensor(val),
                double val => new Tensor(val),
                _ => throw new NotImplementedException("")
            };
            _tensor.SetReferencedByNDArray();

            var _handle = c_api.TFE_NewTensorHandle(_tensor, tf.Status.Handle);
            _tensor.SetEagerTensorHandle(_handle);
        }

        void Init(Array value, Shape? shape = null)
        {
            _tensor = new Tensor(value, shape ?? value.GetShape());
            _tensor.SetReferencedByNDArray();

            var _handle = c_api.TFE_NewTensorHandle(_tensor, tf.Status.Handle);
            _tensor.SetEagerTensorHandle(_handle);
        }

        void Init(Shape shape, TF_DataType dtype = TF_DataType.TF_DOUBLE)
        {
            _tensor = new Tensor(shape, dtype: dtype);
            _tensor.SetReferencedByNDArray();

            var _handle = c_api.TFE_NewTensorHandle(_tensor, tf.Status.Handle);
            _tensor.SetEagerTensorHandle(_handle);
        }

        void Init(Tensor value, Shape? shape = null)
        {
            if (shape is not null)
                _tensor = new Tensor(value.TensorDataPointer, shape, value.dtype);
            else
                _tensor = value;

            if (_tensor.TensorDataPointer == IntPtr.Zero)
                _tensor = tf.get_default_session().eval(_tensor);

            _tensor.SetReferencedByNDArray();

            var _handle = c_api.TFE_NewTensorHandle(_tensor, tf.Status.Handle);
            _tensor.SetEagerTensorHandle(_handle);
        }
    }
}
