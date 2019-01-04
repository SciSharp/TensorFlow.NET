using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using TF_DataType = Tensorflow.DataType;

namespace Tensorflow
{
    /// <summary>
    /// TensorFlow uses a dataflow graph to represent your computation in terms of the dependencies between individual operations. 
    /// This leads to a low-level programming model in which you first define the dataflow graph, 
    /// then create a TensorFlow session to run parts of the graph across a set of local and remote devices.
    /// https://www.tensorflow.org/guide/graphs
    /// </summary>
    public partial class Graph : IDisposable
    {
        private IntPtr _handle;
        private Dictionary<int, Operation> _nodes_by_id;
        private Dictionary<string, Operation> _nodes_by_name;
        private Dictionary<string, int> _names_in_use;
        public int _version;
        private int _next_id_counter;
        private List<String> _unfetchable_ops = new List<string>();

        private string _name_stack;
        public Status Status { get; }

        public Graph()
        {
            _handle = c_api.TF_NewGraph();
            Status = new Status();
        }

        public Graph(IntPtr graph)
        {
            _handle = graph;
            Status = new Status();
            _nodes_by_id = new Dictionary<int, Operation>();
            _nodes_by_name = new Dictionary<string, Operation>();
            _names_in_use = new Dictionary<string, int>();
        }

        public Operation NewOperation(string opType, string opName, Tensor t)
        {
            var desc = c_api.TF_NewOperation(_handle, opType, opName);
            
            c_api.TF_SetAttrTensor(desc, "value", t, Status);
            Status.Check();

            c_api.TF_SetAttrType(desc, "dtype", t.dtype);

            var op = c_api.TF_FinishOperation(desc, Status);
            Status.Check();

            return op;
        }

        public T as_graph_element<T>(T obj, bool allow_tensor = true, bool allow_operation = true)
        {
            return _as_graph_element_locked(obj, allow_tensor, allow_operation);
        }

        private Func<object> _as_graph_element(object obj)
        {
            return null;
        }

        private T _as_graph_element_locked<T>(T obj, bool allow_tensor = true, bool allow_operation = true)
        {
            string types_str = "";

            if (allow_tensor && allow_operation)
            {
                types_str = "Tensor or Operation";
            }
            else if (allow_tensor)
            {
                types_str = "Tensor";
            }
            else if (allow_operation)
            {
                types_str = "Operation";
            }

            var temp_obj = _as_graph_element(obj);

            if(obj is Tensor && allow_tensor)
            {
                if ((obj as Tensor).Graph.Equals(this))
                {
                    return obj;
                }
                else
                {
                    throw new Exception($"Tensor {obj} is not an element of this graph.");
                }
            }

            throw new Exception($"Can not convert a {typeof(T).Name} into a {types_str}.");
        }

        public unsafe Operation create_op(string op_type, List<Tensor> inputs, TF_DataType[] dtypes, 
            TF_DataType[] input_types = null, string name = "", 
            Dictionary<string, AttrValue> attrs = null, OpDef op_def = null)
        {
            if (String.IsNullOrEmpty(name))
            {
                name = op_type;
            }

            name = name.EndsWith("/") ? ops._name_from_scope_name(name) : unique_name(name);
            var node_def = ops._NodeDef(op_type, name, device: "", attrs: attrs);

            var op = new Operation(node_def, 
                this,
                inputs: inputs,
                output_types: dtypes,
                control_inputs: new object[] { },
                input_types: input_types,
                original_op: null,
                op_def: op_def);

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

        public bool is_fetchable<T>(T tensor_or_op)
        {
            if (tensor_or_op is Tensor)
            {
                return !_unfetchable_ops.Contains((tensor_or_op as Tensor).name); ;
            }
            else if (tensor_or_op is Operation)
            {
                return !_unfetchable_ops.Contains((tensor_or_op as Operation).Name);
            }

            return false;
        }

        public string name_scope(string name)
        {
            string new_stack = "";

            if (name.EndsWith("/"))
            {
                new_stack = ops._name_from_scope_name(name);
            }
            else
            {
                new_stack = unique_name(name);
            }

            _name_stack = new_stack;

            return String.IsNullOrEmpty(new_stack) ? "" : new_stack + "/";
        }

        public string unique_name(string name)
        {
            if (!String.IsNullOrEmpty(_name_stack))
            {
                name = _name_stack + "/" + name;
            }

            var name_key = name.ToLower();
            if (_names_in_use.ContainsKey(name_key))
            {
                _names_in_use[name_key]++;
            }
            else
            {
                _names_in_use[name_key] = 1;
                return name;
            }

            return $"{name}_{_names_in_use[name_key]}";
        }

        public TF_Output[] ReturnOutputs(IntPtr results)
        {
            IntPtr return_output_handle = IntPtr.Zero;
            int num_return_outputs = 0;
            c_api.TF_ImportGraphDefResultsReturnOutputs(results, ref num_return_outputs, ref return_output_handle);
            TF_Output[] return_outputs = new TF_Output[num_return_outputs];
            for (int i = 0; i < num_return_outputs; i++)
            {
                var handle = return_output_handle + (Marshal.SizeOf<TF_Output>() * i);
                return_outputs[i] = Marshal.PtrToStructure<TF_Output>(handle);
            }

            return return_outputs;
        }

        public unsafe Operation[] ReturnOperations(IntPtr results)
        {
            TF_Operation return_oper_handle = new TF_Operation();
            int num_return_opers = 0;
            c_api.TF_ImportGraphDefResultsReturnOperations(results, ref num_return_opers, ref return_oper_handle);
            Operation[] return_opers = new Operation[num_return_opers];
            for (int i = 0; i < num_return_opers; i++)
            {
                var handle = return_oper_handle.node + Marshal.SizeOf<TF_Operation>() * i;
                return_opers[i] = new Operation(*(IntPtr*)handle);
            }

            return return_opers;
        }

        public Operation OperationByName(string operName)
        {
            return c_api.TF_GraphOperationByName(_handle, operName);
        }

        public Operation[] get_operations()
        {
            return _nodes_by_name.Values.Select(x => x).ToArray();
        }

        public void Dispose()
        {
            c_api.TF_DeleteGraph(_handle);
        }

        public static implicit operator IntPtr(Graph graph)
        {
            return graph._handle;
        }
    }
}
