using System;
using System.Runtime.CompilerServices;

namespace Tensorflow
{
    public partial class Tensor
    {
        public static Tensor operator !=(Tensor x, int y)
            => gen_math_ops.not_equal(x, math_ops.cast(y, dtype: x.dtype));
        public static Tensor operator ==(Tensor x, int y)
            => gen_math_ops.equal(x, math_ops.cast(y, dtype: x.dtype));
    }
}
