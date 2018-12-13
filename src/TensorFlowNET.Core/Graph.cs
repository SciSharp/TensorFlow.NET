using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using Tensorflow;
using TF_DataType = Tensorflow.DataType;

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
        private Dictionary<string, int> _names_in_use;
        public int _version;
        private int _next_id_counter;

        public Graph(IntPtr graph)
        {
            this.handle = graph;
            _nodes_by_id = new Dictionary<int, Operation>();
            _nodes_by_name = new Dictionary<string, Operation>();
            _names_in_use = new Dictionary<string, int>();
        }

        public unsafe Operation create_op(string op_type, object inputs, TF_DataType[] dtypes, TF_DataType[] input_types = null, Dictionary<string, AttrValue> attrs = null, string name = "Const")
        {
            if (String.IsNullOrEmpty(name))
            {
                name = op_type;
            }

            name = unique_name(name);
            var node_def = ops._NodeDef(op_type, name, device: "", attrs: attrs);

            var op = new Operation(node_def, this, inputs, dtypes);

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

        public string unique_name(string name)
        {
            var name_key = name.ToLower();
            if (_names_in_use.ContainsKey(name_key))
            {
                _names_in_use[name_key]++;
            }
            else
            {
                _names_in_use[name_key] = 1;
            }
                

            return $"{name}_{_names_in_use[name_key]}";
        }

        public Operation[] get_operations()
        {
            return _nodes_by_name.Values.Select(x => x).ToArray();
        }
    }
}
