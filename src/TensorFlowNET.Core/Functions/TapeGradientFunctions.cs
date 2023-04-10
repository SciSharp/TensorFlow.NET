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
        protected EagerDefinedFunction _forward_function;
        protected FuncGraph _forward_graph;
        protected List<int> _forwardprop_output_indices;
        protected int _num_forwardprop_outputs;
        protected ConcreteFunction _backward_function;
        BackwardFunction _backward_function_wrapper;

        public TapeGradientFunctions(FuncGraph func_graph,
            bool need_gradients_for_jvps)
        {
            _func_graph = func_graph;
        }

        public virtual EagerDefinedFunction Forward(Tensors inference_args, Tensors input_tangents = null)
        {
            // TODO(Rinne): add input_tangents arg.
            return ForwardAndBackwardFunctions(inference_args);
        }

        /// <summary>
        /// Record the function call operation.
        /// </summary>
        /// <param name="flat_outputs"></param>
        /// <param name="inference_args"></param>
        public virtual void Record(Tensors flat_outputs, Tensors inference_args)
        {
            // TODO(Rinne): add arg `input_tagents`.
            var (backward_function, to_record) = _wrap_backward_function(_forward_graph, _backward_function, flat_outputs);
            tf.Runner.RecordGradient(_forward_function.Name, inference_args, new object[0], to_record,
                getBackwardFunction: backward_function);
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
            var backward_function_inputs = backward.Inputs.Length - backward.CapturedInputs.Length;
            var recorded_outputs = new Tensors();
            var trainable_recorded_outputs = 0;
            foreach (var (output_index, output) in enumerate(outputs))
            {
                if (trainable_recorded_outputs < backward_function_inputs)
                    recorded_outputs.Add(output);
                if (gradients_util.IsTrainable(output))
                    trainable_recorded_outputs += 1;
            }

            if(_backward_function_wrapper == null)
            {
                var capture_mapping = new Dictionary<long, Tensor>();
                foreach (var (i, output) in enumerate(outputs))
                    capture_mapping[forward_graph.Outputs[i].Id] = output;

                var remapped_captures = new Tensors();
                foreach (var capture in backward.CapturedInputs)
                {
                    if (capture_mapping.ContainsKey(capture.Id))
                        remapped_captures.Add(capture_mapping[capture.Id]);
                }

                var skip_positions = new List<int>();
                foreach (var (output_index, output) in enumerate(outputs))
                {
                    if (!gradients_util.IsTrainable(output))
                        skip_positions.Add(output_index);
                }

                _backward_function_wrapper = (args, unneeded_gradients) =>
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
            }

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

            var backwards_graph = new FuncGraph(_func_graph.Name);
            backwards_graph.as_default();
            var gradients_wrt_outputs = new List<Tensor>();
            foreach (var output in trainable_outputs)
            {
                var (gradient_shape, gradient_dtype) = default_gradient.shape_and_dtype(output);
                var gradient_placeholder = tf.placeholder(gradient_dtype, gradient_shape);
                gradients_wrt_outputs.Add(gradient_placeholder);
                handle_data_util.copy_handle_data(output, gradient_placeholder);
            }
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

            var (forward_function, backward_function) = monomorphic_function_utils._create_forward_backward_with_graph(null, _func_graph, backwards_graph);
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
            
            return (forward_function, _func_graph, backward_function, null, 0);
        }

        public virtual EagerDefinedFunction ForwardAndBackwardFunctions(Tensors inference_args)
        {
            throw new NotImplementedException("");
        }
    }
}
