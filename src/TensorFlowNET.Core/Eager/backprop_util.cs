using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Operations;

namespace Tensorflow.Eager
{
    internal static class backprop_util
    {
        // TODO: add quantized_dtypes (after being supported).
        private static HashSet<TF_DataType> _trainable_dtypes = new HashSet<TF_DataType>(new TF_DataType[]
        {
            dtypes.float16, dtypes.float32, dtypes.float64, dtypes.complex64, dtypes.complex128, 
            dtypes.resource, dtypes.variant, TF_DataType.TF_BFLOAT16
        });
        public static bool IsTrainable(Tensor tensor)
        {
            var dtype = _DTypeFromTensor(tensor);
            return _trainable_dtypes.Contains(dtype);
        }
        public static bool IsTrainable(TF_DataType dtype)
        {
            return _trainable_dtypes.Contains(dtype);
        }

        private static TF_DataType _DTypeFromTensor(Tensor tensor)
        {
            var dtype = tensor.dtype;
            if(dtype.as_base_dtype() == TF_DataType.TF_VARIANT)
            {
                CppShapeInferenceResult.Types.HandleData handle_data;
                if (tensor is EagerTensor)
                {
                    handle_data = tensor.HandleData;
                }
                else
                {
                    handle_data = handle_data_util.get_resource_handle_data(tensor);
                }
                if(handle_data is not null && handle_data.IsSet && handle_data.ShapeAndType is not null &&
                    handle_data.ShapeAndType.Count > 0)
                {
                    var first_type = handle_data.ShapeAndType[0].Dtype;
                    if(first_type != DataType.DtInvalid && handle_data.ShapeAndType.All(x => x.Dtype == first_type))
                    {
                        return first_type.as_tf_dtype();
                    }
                }
            }
            return dtype;
        }
    }
}
