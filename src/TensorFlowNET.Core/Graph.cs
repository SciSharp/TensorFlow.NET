using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace TensorFlowNET.Core
{
    /// <summary>
    /// TensorFlow uses a dataflow graph to represent your computation in terms of the dependencies between individual operations. 
    /// This leads to a low-level programming model in which you first define the dataflow graph, 
    /// then create a TensorFlow session to run parts of the graph across a set of local and remote devices.
    /// https://www.tensorflow.org/guide/graphs
    /// </summary>
    public class Graph
    {
        public IntPtr handle;
        private Dictionary<int, Operation> _nodes_by_id;
        private Dictionary<string, Operation> _nodes_by_name;
        public int _version;
        private int _next_id_counter;

        public Graph(IntPtr graph)
        {
            this.handle = graph;
            _nodes_by_id = new Dictionary<int, Operation>();
            _nodes_by_name = new Dictionary<string, Operation>();
        }

        public unsafe Operation create_op(object inputs, string op_type = "", string name = "")
        {
            if (String.IsNullOrEmpty(name))
            {
                op_type = name;
            }

            var op = new Operation(this, inputs);
            op.name = name;

            return op;
        }

        public void _add_op(Operation op)
        {
            _nodes_by_id[op._id] = op;
            //_nodes_by_name[op.name] = op;
            _version = Math.Max(_version, op._id);
        }

        public int _next_id()
        {
            return ++_next_id_counter;
        }

        public void get_operations()
        {

        }
    }
}
