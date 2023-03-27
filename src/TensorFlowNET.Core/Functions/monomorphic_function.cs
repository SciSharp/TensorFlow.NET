using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Graphs;

namespace Tensorflow.Functions
{
    public class DelayedRewriteGradientFunctions
    {
        static readonly string _INFERENCE_PREFIX = "__inference_";
        static readonly string _BACKWARD_PREFIX = "__backward_";
        static readonly string _FORWARD_PREFIX = "__forward_";
        FuncGraph _func_graph;
        EagerDefinedFunction _inference_function;
        Dictionary<string, string> _attrs;
        int _num_inference_outputs;
        public DelayedRewriteGradientFunctions(FuncGraph func_graph, Dictionary<string, string> attrs)
        {
            _func_graph= func_graph;
            _inference_function = new EagerDefinedFunction(_inference_name(_func_graph.Name), 
                _func_graph, _func_graph.Inputs, _func_graph.Outputs, attrs);
            _attrs = attrs;
            _num_inference_outputs = _func_graph.Outputs.Length;
        }

        public EagerDefinedFunction forward(Tensors inference_args = null, Tensors input_tangents = null)
        {
            if(input_tangents is not null)
            {
                throw new InvalidArgumentError($"unexpectedly got forwardprop information in " +
                    $"a class that does not support forwardprop.");
            }
            return _inference_function;
        }

        private static string _inference_name(string name)
        {
            return $"{_INFERENCE_PREFIX}{name}_{ops.uid()}";
        }
    }
}
