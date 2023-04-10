using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Gradients
{
    internal static class default_gradient
    {
        public static (Shape, TF_DataType) shape_and_dtype(Tensor t)
        {
            if(t.dtype == dtypes.resource)
            {
                var handle_data = resource_variable_ops.get_eager_safe_handle_data(t);
                if(handle_data is null || !handle_data.IsSet || handle_data.ShapeAndType.Count != 1)
                {
                    throw new ValueError($"Internal error: Tried to take gradients (or similar) " +
                        $"of a variable without handle data:\n{t}");
                }
                return (new Shape(handle_data.ShapeAndType[0].Shape), handle_data.ShapeAndType[0].Dtype.as_tf_dtype());
            }
            return (t.shape, t.dtype);
        }

        public static Tensor zeros_like(Tensor t)
        {
            if(t.dtype == dtypes.resource)
            {
                var (shape, dtype) = shape_and_dtype(t);
                return array_ops.zeros(shape, dtype);
            }
            else
            {
                return array_ops.zeros_like(t);
            }
        }

        public static TF_DataType get_zeros_dtype(Tensor t)
        {
            if(t.dtype == dtypes.resource)
            {
                var handle_data = resource_variable_ops.get_eager_safe_handle_data(t);
                if(handle_data is null || !handle_data.IsSet || handle_data.ShapeAndType.Count != 1)
                {
                    throw new ValueError($"Internal error: Tried to take gradients (or similar) " +
                        $"of a variable without handle data:\n{t}");
                }
                return handle_data.ShapeAndType[0].Dtype.as_tf_dtype();
            }
            return t.dtype;
        }
    }
}
