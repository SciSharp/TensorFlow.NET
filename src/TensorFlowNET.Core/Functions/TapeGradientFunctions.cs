using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Graphs;
using static Tensorflow.Binding;
using static Tensorflow.tensorflow;

namespace Tensorflow.Functions
{
    /// <summary>
    /// Caches forward and backward functions compatible with eager gradients.
    /// </summary>
    public abstract class TapeGradientFunctions
    {
        string FORWARD_FUNCTION_ATTRIBUTE_NAME = "forward_function_name";
        string BACKWARD_FUNCTION_ATTRIBUTE_NAME = "backward_function_name";
        string _FORWARD_PREFIX = "__forward_";
        string _BACKWARD_PREFIX = "__backward_";
        string _INFERENCE_PREFIX = "__inference_";

        protected FuncGraph _func_graph;
        protected EagerDefinedFunction _forward;
        protected FuncGraph _forward_graph;
        protected List<int> _forwardprop_output_indices;
        protected int _num_forwardprop_outputs;
        protected ConcreteFunction _backward;

        public TapeGradientFunctions(FuncGraph func_graph,
            bool need_gradients_for_jvps)
        {
            _func_graph = func_graph;
        }

        public EagerDefinedFunction Forward(Tensors inference_args)
        {
            return ForwardAndBackwardFunctions(inference_args);
        }

        /// <summary>
        /// Record the function call operation.
        /// </summary>
        /// <param name="flat_outputs"></param>
        /// <param name="inference_args"></param>
        public void Record(Tensors flat_outputs, Tensors inference_args)
        {
            var (backward_function, to_record) = _wrap_backward_function(_forward_graph, _backward, flat_outputs);
            tf.Runner.RecordGradient(_forward.Name, inference_args, new object[0], to_record,
                getBackwardFunction: () => backward_function);
        }

        (BackwardFunction, Tensors) _wrap_backward_function(FuncGraph forward_graph, ConcreteFunction backward, Tensors flat_outputs)
        {
            BackwardFunction _backward_function_wrapper = (output_grads, unneeded_gradients) =>
            {
                var processed_args = new List<Tensor>();
                var input_index = 0;
                foreach (var (output_index, arg) in enumerate(output_grads))
                {
                    if (arg is null)
                        throw new NotImplementedException("");
                    processed_args.add(arg);
                    input_index += 1;
                }
                return output_grads;// backward.Invoke(processed_args.ToArray());
            };

            return (_backward_function_wrapper, flat_outputs);
        }

        protected (EagerDefinedFunction, FuncGraph, ConcreteFunction, List<int>, int) 
            BuildFunctionsForOutputs(Tensors outputs, Tensors inference_args)
        {
            var trainable_outputs = new List<Tensor>();
            var trainable_indices = new List<int>();
            foreach(var (index, output) in enumerate(outputs))
            {
                if (gradients_util.IsTrainable(output))
                {
                    trainable_outputs.Add(output);
                    trainable_indices.Add(index);
                }
            }

            var gradients_wrt_outputs = new List<Tensor>();
            var backwards_graph = new FuncGraph($"{_BACKWARD_PREFIX}{_func_graph.FuncName}_{ops.uid()}");
            foreach (var output in trainable_outputs)
                gradients_wrt_outputs.Add(tf.placeholder(output.dtype, output.shape));
            var gradients_wrt_inputs = gradients_util._GradientsHelper(trainable_outputs.ToArray(),
                _func_graph.Inputs,
                grad_ys: gradients_wrt_outputs.ToArray(),
                src_graph: _func_graph);

            tf.Context.restore_mode();

            var forward_function_name = $"{_FORWARD_PREFIX}{_func_graph.FuncName}_{ops.uid()}";
            var backward_function_attr = new Dictionary<string, string>();
            backward_function_attr[FORWARD_FUNCTION_ATTRIBUTE_NAME] = forward_function_name;
            backwards_graph.Inputs = gradients_wrt_outputs;
            backwards_graph.Outputs = gradients_wrt_inputs;

            var backward_function = new ConcreteFunction(backwards_graph, backward_function_attr);

            var forward_function_attr = new Dictionary<string, string>();
            forward_function_attr[BACKWARD_FUNCTION_ATTRIBUTE_NAME] = backward_function.Name;
            var forward_function = new EagerDefinedFunction(forward_function_name, _func_graph, 
                _func_graph.Inputs, _func_graph.Outputs, forward_function_attr);
            
            return (forward_function, _func_graph, backward_function, null, 0);
        }

        public virtual EagerDefinedFunction ForwardAndBackwardFunctions(Tensors inference_args)
        {
            throw new NotImplementedException("");
        }
    }
}
