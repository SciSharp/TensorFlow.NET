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
            return constant_op.Constant(nd, name, verify_shape);
        }

        public static Tensor zeros(Shape shape, TF_DataType dtype = TF_DataType.TF_FLOAT, string name = "")
        {
            return array_ops.zeros(shape, dtype, name);
        }
    }
}
