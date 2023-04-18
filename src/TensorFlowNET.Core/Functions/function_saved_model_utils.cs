using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Operations;
using Tensorflow.Train;
using Tensorflow.Variables;
using static Tensorflow.Binding;

namespace Tensorflow.Functions
{
    public static class function_saved_model_utils
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="concrete_function"></param>
        /// <param name="inputs">a list tensors or other objects (such as variables) which 
        /// contain tensors that were originally captured by the function</param>
        public static void restore_captures(ConcreteFunction concrete_function, IEnumerable<object> inputs)
        {
            var bound_inputs = inputs?.Select(obj =>
            {
                if(obj is Tensor tensor)
                {
                    return get_tensor_from_node(tensor);
                }
                else if(obj is IVariableV1 variable)
                {
                    return get_tensor_from_node(variable);
                }
                else
                {
                    throw new TypeError("Encountered an type error, please submit an issue to " +
                        "https://github.com/SciSharp/TensorFlow.NET/issues");
                }
            });
            var bound_variables = inputs.Where(obj => obj is IVariableV1).Select(x => (IVariableV1)x);

            List<Tensor> captured_inputs_list = new();
            concrete_function.set_variables(bound_variables);

            if (bound_inputs is not null)
            {
                foreach(var (bound_input, internal_capture) in zip(bound_inputs, concrete_function.Inputs.Skip(concrete_function.Inputs.Length - bound_inputs.Count())))
                {
                    if(hasattr(bound_input, "__tf_experimental_restore_capture__"))
                    {
                        throw new NotImplementedException();
                    }
                    else
                    {
                        captured_inputs_list.Add(bound_input);
                        concrete_function.func_graph.replace_capture(bound_input, internal_capture);
                        if(internal_capture.dtype == dtypes.resource)
                        {
                            if (resource_variable_ops.is_resource_variable(bound_input))
                            {
                                handle_data_util.copy_handle_data(bound_input.Handle, internal_capture);
                            }
                            else
                            {
                                handle_data_util.copy_handle_data(bound_input, internal_capture);
                            }
                        }
                        concrete_function.func_graph.capture(bound_input);
                    }
                }
            }

            if(captured_inputs_list.Any(inp => inp is null))
            {
                // TODO(Rinne): add warnings.
            }
            concrete_function.SetExternalCaptures(captured_inputs_list);
        }

        public static Tensor get_tensor_from_node(Tensor node)
        {
            return node;
        }
        public static Tensor get_tensor_from_node(IVariableV1 node)
        {
            if (resource_variable_ops.is_resource_variable(node))
            {
                return node.Handle;
            }
            else
            {
                throw new TypeError("Encountered an type error, please submit an issue to " +
                    "https://github.com/SciSharp/TensorFlow.NET/issues");
            }
        }
    }
}
