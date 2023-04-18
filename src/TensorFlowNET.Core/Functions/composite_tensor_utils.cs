using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Framework;
using Tensorflow.Framework.Models;
using Tensorflow.Util;

namespace Tensorflow.Functions
{
    internal static class composite_tensor_utils
    {
        public static List<object> flatten_with_variables(object inputs)
        {
            List<object> flat_inputs = new();
            foreach(var value in nest.flatten(inputs))
            {
                if(value is CompositeTensor && !resource_variable_ops.is_resource_variable(value))
                {
                    throw new NotImplementedException("The composite tensor has not been fully supported.");
                }
                else
                {
                    flat_inputs.Add(value);
                }
            }
            return flat_inputs;
        }
        public static List<object> flatten_with_variables_or_variable_specs(object arg)
        {
            List<object> flat_inputs = new();
            foreach(var value in nest.flatten(arg))
            {
                if(value is CompositeTensor && !resource_variable_ops.is_resource_variable(value))
                {
                    throw new NotImplementedException("The composite tensor has not been fully supported.");
                }
                // TODO(Rinne): deal with `VariableSpec`.
                else if(value is TypeSpec type_spec && value is not TensorSpec)
                {
                    throw new NotImplementedException("The TypeSpec has not been fully supported.");
                }
                else
                {
                    flat_inputs.Add(value);
                }
            }
            return flat_inputs;
        }
    }
}
