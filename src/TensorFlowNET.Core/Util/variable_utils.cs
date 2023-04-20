using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Framework;

namespace Tensorflow.Util
{
    internal static class variable_utils
    {
        public static Tensor[] convert_variables_to_tensors(object[] values)
        {
            return values.Select(x =>
            {
                if (resource_variable_ops.is_resource_variable(x))
                {
                    return ops.convert_to_tensor(x);
                }
                else if (x is CompositeTensor)
                {
                    throw new NotImplementedException("The composite tensor has not been fully supported.");
                }
                else if(x is Tensor tensor)
                {
                    return tensor;
                }
                else
                {
                    throw new TypeError("Currently the output of function to be traced must be `Tensor`.");
                }
            }).ToArray();
        }
    }
}
