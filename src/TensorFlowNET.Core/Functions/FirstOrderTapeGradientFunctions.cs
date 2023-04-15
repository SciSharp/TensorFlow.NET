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

        public override (EagerDefinedFunction, FuncGraph, ConcreteFunction, List<int>, int)
            ForwardAndBackwardFunctions(Tensors inference_args)
        {
            var outputs = _func_graph.Outputs.Take(_num_inference_outputs).ToArray();
            return BuildFunctionsForOutputs(outputs, inference_args);
        }
    }
}
