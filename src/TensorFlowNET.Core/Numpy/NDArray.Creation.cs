using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow.Numpy
{
    public partial class NDArray
    {
        void Initialize(Shape shape, NumpyDType dtype = NumpyDType.Float)
        {
            _tensor = tf.zeros(shape, dtype: dtype.as_tf_dtype());
        }
    }
}
