using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace Tensorflow.Graphs
{
    internal class GraphOverrideGradientContext: ITensorFlowObject
    {
        Graph _graph;
        Dictionary<string, Func<Operation, object[], Tensor[]>> _new_gradient_function_map;
        public GraphOverrideGradientContext(Graph graph, 
            Dictionary<string, Func<Operation, object[], Tensor[]>> new_gradient_function_map)
        {
            _graph = graph;
            _new_gradient_function_map = new_gradient_function_map;
        }

        [DebuggerStepThrough]
        public void __enter__()
        {
            Debug.Assert(_graph._gradient_function_map.Count == 0);
            _graph._gradient_function_map = _new_gradient_function_map;
        }

        [DebuggerStepThrough]
        public void __exit__()
        {
            _graph._gradient_function_map = new Dictionary<string, Func<Operation, object[], Tensor[]>>();
        }

        public void Dispose()
        {

        }
    }
}
