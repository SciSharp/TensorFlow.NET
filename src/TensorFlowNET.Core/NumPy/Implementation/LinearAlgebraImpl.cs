using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow.NumPy
{
    public class LinearAlgebraImpl
    {
        [AutoNumPy]
        public NDArray lstsq(NDArray a, NDArray b, string rcond = "warn")
            => new NDArray(tf.linalg.lstsq(a, b));

        [AutoNumPy]
        public NDArray norm(NDArray a, Axis axis = null)
        {
            if (a.dtype.is_integer())
            {
                var float_a = math_ops.cast(a, dtype: tf.float32);
                return new NDArray(tf.linalg.norm(float_a, axis: axis));
            }
                
            return new NDArray(tf.linalg.norm(a, axis: axis));
        }
    }
}
