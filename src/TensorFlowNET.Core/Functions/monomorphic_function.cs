using Google.Protobuf;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Eager;
using Tensorflow.Framework.Models;
using Tensorflow.Gradients;
using Tensorflow.Graphs;
using Tensorflow.Common.Extensions;
using Tensorflow.Operations;
using Tensorflow.Framework;
using static Tensorflow.Binding;
using System.Diagnostics;

namespace Tensorflow.Functions
{
    internal static class monomorphic_function_utils
    {
        internal static string _FORWARD_PREFIX = "__forward_";
        internal static string _BACKWARD_PREFIX = "__backward_";
        internal static string _INFERENCE_PREFIX = "__inference_";
        internal static string IMPLEMENTS_ATTRIBUTE_NAME = "_implements";
        internal static string FORWARD_FUNCTION_ATTRIBUTE_NAME = "forward_function_name";
        internal static string BACKWARD_FUNCTION_ATTRIBUTE_NAME = "backward_function_name";
        public static string _inference_name(string name)
        {
            return $"{_INFERENCE_PREFIX}{name}_{ops.uid()}";
        }
        public static string _forward_name(string name)
        {
            return $"{_FORWARD_PREFIX}{name}_{ops.uid()}";
        }
        public static string _backward_name(string name)
        {
            return $"{_BACKWARD_PREFIX}{name}_{ops.uid()}";
        }

        public static (EagerDefinedFunction, ConcreteFunction) _create_forward_backward_with_graph(Dictionary<string, AttrValue> attrs, 
            FuncGraph forward_graph, FuncGraph backwards_graph)
        {
            string forward_function_name = _forward_name(forward_graph.Name);
            Dictionary<string, AttrValue> common_attributes;
            if(attrs is null)
            {
                common_attributes = new Dictionary<string, AttrValue>();
            }
            else
            {
                common_attributes = new Dictionary<string, AttrValue>(attrs);
            }

            if (common_attributes.ContainsKey(IMPLEMENTS_ATTRIBUTE_NAME))
            {
                common_attributes.Remove(IMPLEMENTS_ATTRIBUTE_NAME);
            }
            var backward_function_attr = _parse_func_attrs(new Dictionary<string, object>()
            {
                {FORWARD_FUNCTION_ATTRIBUTE_NAME, forward_function_name }
            });
            backward_function_attr.Update(common_attributes);
            var backward_function = new ConcreteFunction(backwards_graph, backward_function_attr);
            var forward_function_attr = _parse_func_attrs(new Dictionary<string, object>()
            {
                {BACKWARD_FUNCTION_ATTRIBUTE_NAME, backward_function.Name }
            });
            forward_function_attr.Update(common_attributes);
            var forward_function = new EagerDefinedFunction(forward_function_name, forward_graph,
                forward_graph.Inputs, forward_graph.Outputs, forward_function_attr);
            return (forward_function, backward_function);
        }

        public static Dictionary<string, AttrValue> _parse_func_attrs(Dictionary<string, object> attributes)
        {
            Dictionary<string, AttrValue> attrs = new();
            foreach(var item in attributes)
            {
                var key = item.Key;
                var value = item.Value;
                if (value is AttrValue attr_value)
                {
                    attrs[key] = attr_value;
                }
                else if (value is bool b)
                {
                    attrs[key] = new AttrValue() { B = b };
                }
                else if (value is int i)
                {
                    attrs[key] = new AttrValue() { I = i };
                }
                else if (value is float f)
                {
                    attrs[key] = new AttrValue() { F = f };
                }
                else if(value is string s)
                {
                    attrs[key] = new AttrValue() { S = ByteString.CopyFromUtf8(s) };
                }
                else if (value is byte[] bytes)
                {
                    attrs[key] = new AttrValue() { S = ByteString.CopyFrom(bytes) };
                }
                else
                {
                    throw new ValueError($"Attribute {key} must be bool, int, float, string, or " +
                        $"AttrValue. Got {value.GetType()}.");
                }
            }
            return attrs;
        }

        public static Dictionary<string, AttrValue> _parse_func_attrs(Dictionary<string, string> attributes)
        {
            Dictionary<string, AttrValue> attrs = new();
            foreach (var item in attributes)
            {
                var key = item.Key;
                var value = item.Value;
                attrs[key] = new AttrValue() { S = ByteString.CopyFromUtf8(value) };
            }
            return attrs;
        }
    }
    public class DelayedRewriteGradientFunctions : TapeGradientFunctions
    {
        EagerDefinedFunction _inference_function;
        Dictionary<string, AttrValue> _attrs;
        int _num_inference_outputs;
        Dictionary<int, (EagerDefinedFunction, ConcreteFunction)> _cached_function_pairs = new();
        public DelayedRewriteGradientFunctions(FuncGraph func_graph, Dictionary<string, AttrValue> attrs)
            : base(func_graph, false)
        {
            _func_graph = func_graph;
            _inference_function = new EagerDefinedFunction(monomorphic_function_utils._inference_name(_func_graph.Name),
                _func_graph, _func_graph.Inputs, _func_graph.Outputs, attrs);
            _attrs = attrs;
            _num_inference_outputs = _func_graph.Outputs.Length;
        }

        public override EagerDefinedFunction Forward(Tensors inference_args = null, Tensors input_tangents = null)
        {
            if (input_tangents is not null)
            {
                throw new InvalidArgumentError($"unexpectedly got forwardprop information in " +
                    $"a class that does not support forwardprop.");
            }
            return _inference_function;
        }

        public override void Record(Tensors flat_outputs, Tensors inference_args)
        {
            var (backward_function, to_record) = _backward(flat_outputs);
            foreach(var tape in tf.GetTapeSet())
            {
                tape.RecordOperation(_inference_function.Signature.Name, to_record, 
                    inference_args, backward_function);
            }
        }

        public (EagerDefinedFunction, ConcreteFunction) forward_backward(int num_doutputs = -2)
        {
            if(num_doutputs == -2)
            {
                num_doutputs = _num_inference_outputs;
            }
            if(_cached_function_pairs.TryGetValue(num_doutputs, out var target))
            {
                return target;
            }
            var (forward, backward) = _construct_forward_backward(num_doutputs);
            _cached_function_pairs[num_doutputs] = (forward, backward);
            return (forward, backward);

        }

        private (BackwardFunction, Tensors) _backward(Tensors outputs)
        {
            Tensor[] backward_function(Tensor[] args, long[] unneeded_gradients)
            {
                var call_op = outputs[0].op;
                return _rewrite_forward_and_call_backward(call_op, args);
            }
            return (backward_function, outputs);
        }

        internal Tensor[] _rewrite_forward_and_call_backward(Operation op, params object[] doutputs)
        {
            var (forward_function, backward_function) = forward_backward(doutputs.Length);
            if(backward_function.Outputs is null || backward_function.Outputs.Length == 0)
            {
                return backward_function.FlatStructuredOutputs;
            }
            forward_function.AddToGraph(op.graph);

            op._set_func_attr("f", forward_function.Name);
            op._set_type_list_attr("Tout", forward_function.OutputTypes);
            op._add_outputs(forward_function.OutputTypes.Select(x => x.as_tf_dtype()).
                Skip(op.outputs.Length).ToArray(), forward_function.OutputShapes.Skip(op.outputs.Length).ToArray()
                );
            for(int i = 0; i < op.outputs.Length; i++)
            {
                var func_graph_output = forward_function._func_graph_outputs[i];
                handle_data_util.copy_handle_data(func_graph_output, op.outputs[i]);
            }

            var capture_mapping = zip(_func_graph.Outputs.Select(t => ops.tensor_id(t)), op.outputs).
                ToDictionary(x => x.Item1, x => x.Item2);
            var remapped_captures = backward_function.CapturedInputs.Select(
                x => capture_mapping.GetOrDefault(ops.tensor_id(x), x)
                );

            List<Tensor> cleaned_doutputs = new();
            foreach(var (doutput, placeholder) in zip(doutputs, _func_graph.Outputs))
            {
                if (backprop_util.IsTrainable(placeholder))
                {
                    if(doutput is IndexedSlices)
                    {
                        cleaned_doutputs.Add(ops.convert_to_tensor(doutput));
                    }
                    else if(doutput is null)
                    {
                        cleaned_doutputs.Add(default_gradient.zeros_like(placeholder));
                    }
                    else if(doutput is Tensor tensor)
                    {
                        cleaned_doutputs.Add(tensor);
                    }
                    else
                    {
                        throw new ValueError($"Unsupported type {doutput.GetType()} in function _rewrite_forward_and_call_backward");
                    }
                }
            }

            return backward_function.CallFlat(cleaned_doutputs.ToArray(), remapped_captures.ToArray());
        }

        private (EagerDefinedFunction, ConcreteFunction) _construct_forward_backward(int num_doutputs)
        {
            var trainable_outputs = _func_graph.Outputs.Take(num_doutputs).Where(x => backprop_util.IsTrainable(x));

            List<TensorSpec> signature = new();
            foreach(var t in trainable_outputs)
            {
                var (shape, dtype) = default_gradient.shape_and_dtype(t);
                signature.Add(new TensorSpec(shape, dtype));
            }

            Tensor[] _backprop_function(Tensor[] grad_ys)
            {
                return gradients_util._GradientsHelper(trainable_outputs.ToArray(), _func_graph.Inputs,
                    grad_ys, src_graph: _func_graph);
            }

            _func_graph.as_default();
            FuncGraph backwards_graph = new(monomorphic_function_utils._backward_name(_func_graph.Name));
            FuncGraph.func_graph_from_func(backwards_graph.Name,  x => _backprop_function(x.Select(y =>
            {
                Debug.Assert(y is Tensor);
                return (Tensor)y;
            }).ToArray()), new object[0], new Dictionary<string, object>(), signature.ToArray(), backwards_graph);
            var backwards_graph_captures = backwards_graph.external_captures;
            var captures_from_forward = backwards_graph_captures.Where(c => c is not EagerTensor && c.graph == _func_graph);
            
            HashSet<Tensor> existing_outputs = new HashSet<Tensor>(_func_graph.Outputs);
            foreach(var capture in captures_from_forward)
            {
                if (!existing_outputs.Contains(capture))
                {
                    existing_outputs.Add(capture);
                    _func_graph.Outputs.Add(capture);
                }
            }

            var (forward_function, backward_function) = monomorphic_function_utils._create_forward_backward_with_graph(
                _attrs, _func_graph, backwards_graph);
            _func_graph.Exit();
            return (forward_function, backward_function);
        }
    }
}
