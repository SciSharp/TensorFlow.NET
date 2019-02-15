using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public static partial class tf
    {
        public static unsafe Tensor matmul(Tensor a, Tensor b) => gen_math_ops.mat_mul(a, b);
    }
}
