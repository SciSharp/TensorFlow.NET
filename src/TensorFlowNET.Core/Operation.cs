using System;
using System.Collections.Generic;
using System.Text;

namespace TensorFlowNET.Core
{
    public class Operation
    {
        private Graph _graph;
        private IntPtr _c_op;

        public Operation(Graph g, object inputs)
        {
            _graph = g;

            _c_op = ops._create_c_op(g, inputs);
        }
    }
}
