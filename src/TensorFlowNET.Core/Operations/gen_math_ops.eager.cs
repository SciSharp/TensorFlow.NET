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
        public static EagerTensor mul(IntPtr x, IntPtr y, string name = null)
        {
            var results = EagerTensorPass.Create();
            using Status status = new Status(c_api.TFE_FastPathExecute(tf.context, tf.context.device_name,
                "Mul", name, new IntPtr[]
                {
                    x,
                    y,
                }, 2, 
                null, null,
                results.Points, results.Length));
            status.Check(true);
            return results[0].Resolve();
        }
    }
}
