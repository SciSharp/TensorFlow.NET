using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public static partial class tf
    {
        public static Tensor constant(NDArray nd, string name = "Const", bool verify_shape = false)
        {
            return constant_op.Create(nd, name, verify_shape);
        }
    }
}
