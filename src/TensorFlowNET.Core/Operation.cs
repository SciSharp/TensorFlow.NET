using System;
using System.Collections.Generic;
using System.Text;

namespace TensorFlowNET.Core
{
    public class Operation
    {
        private Graph _graph;

        public Operation(Graph g)
        {
            _graph = g;
        }
    }
}
