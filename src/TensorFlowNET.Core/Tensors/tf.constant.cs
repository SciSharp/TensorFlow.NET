using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public static partial class tf
    {
        public static Tensor constant(NDArray nd, string name = "Const") => constant_op.constant(nd, name: name);

        public static Tensor zeros(Shape shape, TF_DataType dtype = TF_DataType.TF_FLOAT, string name = "") => array_ops.zeros(shape, dtype, name);
    }
}
