using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Graphs;

namespace Tensorflow.Functions
{
    public class DelayedRewriteGradientFunctions: TapeGradientFunctions
    {
        EagerDefinedFunction _inference_function;
        Dictionary<string, string> _attrs;
        int _num_inference_outputs;
        public DelayedRewriteGradientFunctions(FuncGraph func_graph, Dictionary<string, string> attrs)
            :base(func_graph, false)
        {
            _func_graph= func_graph;
            _inference_function = new EagerDefinedFunction(_inference_name(_func_graph.Name), 
                _func_graph, _func_graph.Inputs, _func_graph.Outputs, attrs);
            _attrs = attrs;
            _num_inference_outputs = _func_graph.Outputs.Length;
        }

        public override EagerDefinedFunction Forward(Tensors inference_args = null, Tensors input_tangents = null)
        {
            if(input_tangents is not null)
            {
                throw new InvalidArgumentError($"unexpectedly got forwardprop information in " +
                    $"a class that does not support forwardprop.");
            }
            return _inference_function;
        }

        public override void Record(Tensors flat_outputs, Tensors inference_args)
        {
            // TODO(Rinne): implement it.
            throw new NotImplementedException();
            base.Record(flat_outputs, inference_args);
        }

        //private (BackwardFunction, Tensors) _backward(Tensors outputs)
        //{
        //    Tensor[] backward_function(Tensor[] grads, long[] unneeded_gradients)
        //    {
        //        var call_op = outputs[0].op;

        //    }
        //}

        private string _inference_name(string name)
        {
            return $"{_INFERENCE_PREFIX}{name}_{ops.uid()}";
        }
    }
}
