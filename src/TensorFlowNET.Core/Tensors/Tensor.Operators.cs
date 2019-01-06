using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public partial class Tensor
    {
        public static Tensor operator +(Tensor t1, Tensor t2)
        {
            return gen_math_ops.add(t1, t2);
        }
    }
}
