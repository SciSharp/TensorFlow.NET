using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public static partial class tf
    {
        public static unsafe Tensor add(Tensor a, Tensor b)
        {
            return gen_math_ops.add(a, b);
        }

        public static unsafe Tensor multiply(Tensor x, Tensor y)
        {
            return gen_math_ops.mul(x, y);
        }

        public static unsafe Tensor multiply(Tensor x, RefVariable y)
        {
            return gen_math_ops.mul(x, y);
        }
    }
}
