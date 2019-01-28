using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class array_ops
    {
        public static Tensor zeros(Shape shape, TF_DataType dtype = TF_DataType.TF_FLOAT, string name = "")
        {
            Tensor output = null;

            dtype = dtype.as_base_dtype();
            Python.with(new ops.name_scope(name, "zeros", shape), self =>
            {
                name = self as ops.name_scope;
                switch (dtype)
                {
                    case TF_DataType.TF_BOOL:
                        output = _constant_if_small(false, shape, dtype, name);
                        break;
                    case TF_DataType.TF_DOUBLE:
                        output = _constant_if_small(0.0D, shape, dtype, name);
                        break;
                    case TF_DataType.TF_FLOAT:
                        output = _constant_if_small(0.0F, shape, dtype, name);
                        break;
                    case TF_DataType.TF_INT32:
                        output = _constant_if_small(0, shape, dtype, name);
                        break;
                    default:
                        break;
                }
            });

            return output;
        }

        private static Tensor _constant_if_small<T>(T value, Shape shape, TF_DataType dtype, string name)
        {
            Tensor tShape = null;
            var nd = np.zeros<T>(shape);
            if (shape.Size < 1000)
            {
                return constant_op.Constant(nd, name);
            }
            else
            {
                tShape = constant_op._tensor_shape_tensor_conversion_function(shape.as_shape());
                var c = constant_op.Constant(0);
                return gen_array_ops.fill(tShape, c, name);
            }
        }
    }
}
