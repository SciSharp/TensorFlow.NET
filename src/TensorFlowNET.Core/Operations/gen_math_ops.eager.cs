using System;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public static partial class gen_math_ops
    {
        public static Tensor mul(IntPtr x, IntPtr y, string name = null)
            => tf.Context.ExecuteOp("Mul", name, new ExecuteOpArgs(x, y));
    }
}
