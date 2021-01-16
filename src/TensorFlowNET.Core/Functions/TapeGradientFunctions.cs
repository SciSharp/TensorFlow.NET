using System;
using System.Collections.Generic;
using System.Linq;
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

        /// <summary>
        /// Create a backward function given `outputs` from the forward function.
        /// </summary>
        /// <param name="forward_graph"></param>
        /// <param name="backward"></param>
        /// <param name="outputs"></param>
        /// <returns></returns>
        (BackwardFunction, Tensors) _wrap_backward_function(FuncGraph forward_graph, ConcreteFunction backward, Tensors outputs)
        {
            var capture_mapping = new Dictionary<long, Tensor>();
            foreach(var (i, output) in enumerate(outputs))
                capture_mapping[forward_graph.Outputs[i].Id] = output;

            var remapped_captures = new Tensors();
            foreach(var capture in backward.CapturedInputs)
            {
                if (capture_mapping.ContainsKey(capture.Id))
                    remapped_captures.Add(capture_mapping[capture.Id]);
            }

            var backward_function_inputs = backward.Inputs.Length - backward.CapturedInputs.Length;
            var recorded_outputs = new Tensors();
            var relevant_outputs = outputs;
            var trainable_recorded_outputs = 0;
            var skip_positions = new List<int>();
            foreach (var (output_index, output) in enumerate(relevant_outputs))
            {
                if (trainable_recorded_outputs < backward_function_inputs)
                    recorded_outputs.Add(output);
                if (gradients_util.IsTrainable(output))
                    trainable_recorded_outputs += 1;
                else
                    skip_positions.Add(output_index);
            }

            BackwardFunction _backward_function_wrapper = (args, unneeded_gradients) =>
            {
                var processed_args = new Tensors();
                var input_index = 0;
                foreach (var (output_index, arg) in enumerate(args))
                {
                    if (skip_positions.Contains(output_index))
                        continue;
                    if (arg == null)
                        throw new NotImplementedException("");
                    processed_args.Add(arg);
                    input_index += 1;
                    if (input_index >= backward_function_inputs)
                        break;
                }

                tf.Logger.Debug($"Invoke backward function: {backward.Name}");
                var gradients = backward.CallFlat(processed_args, remapped_captures);

                foreach (var unneeded_gradient_index in unneeded_gradients)
                {
                    var index = Convert.ToInt32(unneeded_gradient_index);
                    if (gradients.Length <= index)
                        gradients.Insert(index, null);
                }

                return gradients;
            };

            return (_backward_function_wrapper, recorded_outputs);
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
            var backwards_graph = new FuncGraph($"{_BACKWARD_PREFIX}_{_func_graph.FuncName}_{ops.uid()}");
            backwards_graph.as_default();
            foreach (var output in trainable_outputs)
                gradients_wrt_outputs.Add(tf.placeholder(output.dtype, output.shape));
            var gradients_wrt_inputs = gradients_util._GradientsHelper(trainable_outputs.ToArray(),
                _func_graph.Inputs,
                grad_ys: gradients_wrt_outputs.ToArray(),
                src_graph: _func_graph);

            var captures_from_forward = backwards_graph.external_captures
                .Where(x => !x.IsEagerTensor && x.graph == _func_graph)
                .ToArray();
            foreach(var capture in captures_from_forward)
            {
                if (!_func_graph.Outputs.Contains(capture))
                    _func_graph.Outputs.Add(capture);
            }
            backwards_graph.Exit();

            var forward_function_name = $"{_FORWARD_PREFIX}_{_func_graph.FuncName}_{ops.uid()}";
            var backward_function_attr = new Dictionary<string, string>();
            backward_function_attr[FORWARD_FUNCTION_ATTRIBUTE_NAME] = forward_function_name;
            gradients_wrt_outputs.append(backwards_graph.internal_captures);
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
