using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Eager;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public static partial class gen_math_ops
    {
        public static Tensor mul(IntPtr x, IntPtr y, string name = null)
        {
            var results = tf.Runner.TFE_FastPathExecute(tf.context, tf.context.device_name,
                "Mul", name,
                null,
                x, y);

            return results[0];
        }
    }
}
