using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Graphs;

namespace Tensorflow.Functions
{
    public class FirstOrderTapeGradientFunctions : TapeGradientFunctions
    {
        public FirstOrderTapeGradientFunctions(FuncGraph func_graph,
            bool need_gradients_for_jvps) : base(func_graph, 
                need_gradients_for_jvps)
        {

        }

        public override EagerDefinedFunction ForwardAndBackwardFunctions(Tensors inference_args)
        {
            var outputs = _func_graph.Outputs;
            (_forward, _forward_graph, _backward, _forwardprop_output_indices, _num_forwardprop_outputs) 
                = BuildFunctionsForOutputs(outputs, inference_args);
            return _forward;
        }
    }
}
