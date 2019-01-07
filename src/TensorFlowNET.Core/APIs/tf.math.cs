using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public static partial class tf
    {
        public static Tensor add(Tensor a, Tensor b)
        {
            return gen_math_ops.add(a, b);
        }

        public static Tensor sub(Tensor a, Tensor b)
        {
            return gen_math_ops.sub(a, b);
        }

        public static Tensor multiply(Tensor x, Tensor y)
        {
            return gen_math_ops.mul(x, y);
        }

        public static Tensor pow(Tensor x, double y)
        {
            return gen_math_ops.pow(x, y);
        }

        public static Tensor reduce_sum(Tensor input, int[] axis = null)
        {
            return gen_math_ops.sum(input, axis);
        }
    }
}
