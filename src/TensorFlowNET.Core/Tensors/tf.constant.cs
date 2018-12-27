using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public static partial class tf
    {
        public static Tensor constant(object value, TF_DataType dtype = TF_DataType.DtInvalid, TensorShape shape = null, string name = "Const", bool verify_shape = false)
        {
            return constant_op.Create(value, dtype, shape, name, verify_shape);
        }
    }
}
