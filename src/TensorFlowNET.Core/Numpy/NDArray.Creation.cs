using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Eager;
using static Tensorflow.Binding;

namespace Tensorflow.NumPy
{
    public partial class NDArray
    {
        public NDArray(bool value) => _tensor = new EagerTensor(value);
        public NDArray(byte value) => _tensor = new EagerTensor(value);
        public NDArray(short value) => _tensor = new EagerTensor(value);
        public NDArray(int value) => _tensor = new EagerTensor(value);
        public NDArray(long value) => _tensor = new EagerTensor(value);
        public NDArray(float value) => _tensor = new EagerTensor(value);
        public NDArray(double value) => _tensor = new EagerTensor(value);

        public NDArray(Array value, Shape? shape = null) => _tensor = new EagerTensor(value, shape);

        public NDArray(Shape shape, NumpyDType dtype = NumpyDType.Float)
        {
            Initialize(shape, dtype: dtype);
        }

        public NDArray(Tensor value, Shape? shape = null)
        {
            if (shape is not null)
                _tensor = tf.reshape(value, shape);
            else
                _tensor = value;

            if (_tensor.TensorDataPointer == IntPtr.Zero)
                _tensor = tf.get_default_session().eval(_tensor);
        }

        public static NDArray Scalar<T>(T value) where T : unmanaged
        {
            return value switch
            {
                bool val => new NDArray(val),
                int val => new NDArray(val),
                float val => new NDArray(val),
                double val => new NDArray(val),
                _ => throw new NotImplementedException("")
            };
        }

        void Initialize(Shape shape, NumpyDType dtype = NumpyDType.Float)
        {
            // _tensor = tf.zeros(shape, dtype: dtype.as_tf_dtype());
            _tensor = new EagerTensor(shape, dtype: dtype.as_tf_dtype());
        }
    }
}
