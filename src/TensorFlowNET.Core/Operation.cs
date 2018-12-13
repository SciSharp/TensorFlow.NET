using System;
using System.Collections.Generic;
using System.Text;

namespace TensorFlowNET.Core
{
    public class Operation
    {
        private Graph _graph;
        private IntPtr _c_op;
        public int _id => _id_value;
        private int _id_value;
        public string name;

        public Operation(Graph g, object inputs)
        {
            _graph = g;

            _id_value = _graph._next_id();
            _c_op = ops._create_c_op(g, inputs);
            _graph._add_op(this);
        }
    }
}
