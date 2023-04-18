using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Eager;
using Tensorflow.Gradients;
using Tensorflow.Graphs;
using Tensorflow.NumPy;
using Tensorflow.Operations;
using static Tensorflow.Binding;
using static Tensorflow.tensorflow;

namespace Tensorflow.Functions
{
    /// <summary>
    /// Caches forward and backward functions compatible with eager gradients.
    /// </summary>
    public abstract class TapeGradientFunctions
    {
        protected string FORWARD_FUNCTION_ATTRIBUTE_NAME = "forward_function_name";
        protected string BACKWARD_FUNCTION_ATTRIBUTE_NAME = "backward_function_name";
        protected string _FORWARD_PREFIX = "__forward_";
        protected string _BACKWARD_PREFIX = "__backward_";
        protected string _INFERENCE_PREFIX = "__inference_";

        protected FuncGraph _func_graph;
        protected EagerDefinedFunction _forward;
        protected FuncGraph _forward_graph;
        protected List<int> _forwardprop_input_indices;
        protected List<int> _forwardprop_output_indices;
        protected int _num_forwardprop_outputs;
        protected int _num_inference_outputs;
        protected int _num_outputs;
        protected int _num_trainable_inference_outputs;
        protected ConcreteFunction _backward;
        BackwardFunction _backward_function_wrapper;

        public TapeGradientFunctions(FuncGraph func_graph,
            bool need_gradients_for_jvps)
        {
            _func_graph = func_graph;
            _forward_graph = null;
            _forward = null;
            _backward = null;
            _num_outputs = func_graph.Outputs.Length;
            _forwardprop_output_indices = null;
            _num_forwardprop_outputs = 0;
            _num_inference_outputs = func_graph.Outputs.Length;
            _num_trainable_inference_outputs = func_graph.Outputs.Where(t => backprop_util.IsTrainable(t)).Count();
        }

        public virtual EagerDefinedFunction Forward(Tensors inference_args, Tensors input_tangents = null)
        {
            // TODO(Rinne): add input_tangents arg.
            if(_forward is null)
            {
                (_forward, _forward_graph, _backward, _forwardprop_output_indices, _num_forwardprop_outputs)
                 = ForwardAndBackwardFunctions(inference_args);
            }
            return _forward;
        }

        /// <summary>
        /// Record the function call operation.
        /// </summary>
        /// <param name="flat_outputs"></param>
        /// <param name="inference_args"></param>
        public virtual void Record(Tensors flat_outputs, Tensors inference_args)
        {
            // TODO(Rinne): add arg `input_tagents`.
            var (backward_function, to_record) = _wrap_backward_function(_forward_graph, _backward, flat_outputs);
            if(_forwardprop_output_indices is not null && _forwardprop_output_indices.Count > 0)
            {
                // TODO(Rinne): implement it.
                throw new NotImplementedException();
            }
            tf.Runner.TFE_TapeSetRecordOperation(_forward.Signature.Name, to_record, inference_args, backward_function);
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
            var capture_mapping = zip(forward_graph.Outputs.Select(t => ops.tensor_id(t)), outputs)
                .ToDictionary(x => x.Item1, x => x.Item2);
            var captured_inputs = backward.CapturedInputs;
            var remapped_captures = captured_inputs.Select(c =>
            {
                if (capture_mapping.TryGetValue(ops.tensor_id(c), out var value))
                {
                    return value;
                }
                else
                {
                    return c;
                }
            }).ToArray();
            if(remapped_captures.Where(t => t is not EagerTensor).Any(t => t.graph == forward_graph))
            {
                var incorrect_mapping = remapped_captures.Where(t => t is not EagerTensor && t.graph != forward_graph);
                throw new RuntimeError($"Failed to map all backward graph captures to " +
                    $"the forward graph. Incorrectly mapped: {string.Join(", ", incorrect_mapping)}");
            }

            Dictionary<int, Tensor> variant_zeros_like = new Dictionary<int, Tensor>();
            var backward_function_inputs = backward.Inputs.Length - backward.CapturedInputs.Length;
            var recorded_outputs = new Tensors();
            int trainable_recorded_outputs = 0;
            var skip_positions = new HashSet<int>();
            var relevant_outputs = outputs;
            foreach (var (output_index, output) in enumerate(relevant_outputs))
            {
                if (trainable_recorded_outputs < backward_function_inputs)
                    recorded_outputs.Add(output);
                if (backprop_util.IsTrainable(output))
                    trainable_recorded_outputs++;
                else
                    skip_positions.Add(output_index);
                if (output.dtype == dtypes.variant)
                    variant_zeros_like[output_index] = default_gradient.zeros_like(output);
            }

            _backward_function_wrapper = (args, unneeded_gradients) =>
            {
                if(backward.Outputs is null || backward.Outputs.Length == 0)
                {
                    return backward.FlatStructuredOutputs;
                }

                var processed_args = new Tensors();
                int input_index = 0;
                foreach (var (output_index, arg) in enumerate(args))
                {
                    if (skip_positions.Contains(output_index))
                        continue;
                    if (arg is null)
                    {
                        var input_placeholder = backward.Inputs[input_index];
                        Tensor variant_arg;
                        if (input_placeholder.dtype == dtypes.variant)
                        {
                            variant_arg = variant_zeros_like[output_index];
                        }
                        else
                        {
                            var (shape, type) = default_gradient.shape_and_dtype(input_placeholder);

                            variant_arg = array_ops.zeros(shape, type);
                        }
                        processed_args.Add(variant_arg);
                    }
                    else
                    {
                        processed_args.Add(arg);
                    }
                    input_index++;
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
                if (backprop_util.IsTrainable(output))
                {
                    trainable_outputs.Add(output);
                    trainable_indices.Add(index);
                }
            }

            var backwards_graph = new FuncGraph(monomorphic_function_utils._backward_name(_func_graph.Name));
            backwards_graph.as_default();
            var gradients_wrt_outputs = new List<Tensor>();
            foreach (var output in trainable_outputs)
            {
                var (gradient_shape, gradient_dtype) = default_gradient.shape_and_dtype(output);
                var gradient_placeholder = tf.placeholder(gradient_dtype, gradient_shape);
                gradients_wrt_outputs.Add(gradient_placeholder);
                handle_data_util.copy_handle_data(output, gradient_placeholder);
            }
            // TODO(Rinne): with ops.device(None)
            var gradients_wrt_inputs = gradients_util._GradientsHelper(trainable_outputs.ToArray(),
                    _func_graph.Inputs,
                    grad_ys: gradients_wrt_outputs.ToArray(),
                    src_graph: _func_graph);

            var captures_from_forward = backwards_graph.external_captures
                .Where(x => x is not EagerTensor && x is not NDArray && x.graph == _func_graph)
                .ToArray();
            HashSet<Tensor> existing_outputs = new(_func_graph.Outputs);
            foreach(var capture in captures_from_forward)
            {
                if (!existing_outputs.Contains(capture))
                {
                    existing_outputs.Add(capture);
                    _func_graph.Outputs.Add(capture);
                }
            }
            backwards_graph.Exit();

            backwards_graph.Inputs = gradients_wrt_outputs.Concat(backwards_graph.internal_captures).ToArray();
            backwards_graph.Outputs.AddRange(gradients_wrt_inputs.Where(x => x is not null));

            var (wrapped_forward_function, wrapped_backward_function) = 
                monomorphic_function_utils._create_forward_backward_with_graph(null, _func_graph, backwards_graph);
            //var forward_function_name = $"{_FORWARD_PREFIX}_{_func_graph.FuncName}_{ops.uid()}";
            //var backward_function_attr = new Dictionary<string, string>();
            //backward_function_attr[FORWARD_FUNCTION_ATTRIBUTE_NAME] = forward_function_name;

            //var backward_function = new ConcreteFunction(backwards_graph, 
            //    monomorphic_function_utils._parse_func_attrs(backward_function_attr));
            
            //var forward_function_attr = new Dictionary<string, string>();
            //forward_function_attr[BACKWARD_FUNCTION_ATTRIBUTE_NAME] = backward_function.Name;
            //var forward_function = new EagerDefinedFunction(forward_function_name, _func_graph, 
            //    _func_graph.Inputs, _func_graph.Outputs, 
            //    monomorphic_function_utils._parse_func_attrs(forward_function_attr));
            
            return (wrapped_forward_function, _func_graph, wrapped_backward_function, null, 0);
        }

        public virtual (EagerDefinedFunction, FuncGraph, ConcreteFunction, List<int>, int) 
            ForwardAndBackwardFunctions(Tensors inference_args)
        {
            throw new NotImplementedException("");
        }
    }
}
