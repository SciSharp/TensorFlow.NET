using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    /// <summary>
    /// TensorFlow uses a dataflow graph to represent your computation in terms of the dependencies between individual operations. 
    /// This leads to a low-level programming model in which you first define the dataflow graph, 
    /// then create a TensorFlow session to run parts of the graph across a set of local and remote devices.
    /// https://www.tensorflow.org/guide/graphs
    /// </summary>
    public partial class Graph : IPython, IDisposable
    {
        private IntPtr _handle;
        private Dictionary<int, ITensorOrOperation> _nodes_by_id;
        public Dictionary<string, ITensorOrOperation> _nodes_by_name;
        private Dictionary<string, int> _names_in_use;
        public int _version;
        private int _next_id_counter;
        private List<String> _unfetchable_ops = new List<string>();

        public string _name_stack = "";
        public string _graph_key;
        public Status Status { get; }

        /// <summary>
        /// True if the graph is considered "finalized".  In that case no
        /// new operations can be added.
        /// </summary>
        private bool _finalized = false;

        /// <summary>
        /// Arbitrary collections of objects.
        /// </summary>
        private Dictionary<string, object> _collections = new Dictionary<string, object>();

        public Graph()
        {
            _handle = c_api.TF_NewGraph();
            Status = new Status();
            _nodes_by_id = new Dictionary<int, ITensorOrOperation>();
            _nodes_by_name = new Dictionary<string, ITensorOrOperation>();
            _names_in_use = new Dictionary<string, int>();
            _graph_key = $"grap-key-{ops.uid()}/";
        }

        public Graph(IntPtr handle)
        {
            _handle = handle;
            Status = new Status();
            _nodes_by_id = new Dictionary<int, ITensorOrOperation>();
            _nodes_by_name = new Dictionary<string, ITensorOrOperation>();
            _names_in_use = new Dictionary<string, int>();
            _graph_key = $"grap-key-{ops.uid()}/";
        }

        public ITensorOrOperation as_graph_element(object obj, bool allow_tensor = true, bool allow_operation = true)
        {
            return _as_graph_element_locked(obj, allow_tensor, allow_operation);
        }

        public Graph as_default() => ops.set_default_graph(this);

        private Tensor _as_graph_element(object obj)
        {
            if (obj is RefVariable var)
                return var._as_graph_element();

            return null;
        }
       
        private ITensorOrOperation _as_graph_element_locked(object obj, bool allow_tensor = true, bool allow_operation = true)
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
            if (temp_obj != null)
                obj = temp_obj;

            // If obj appears to be a name...
            if (obj is string name)
            {
                if(name.Contains(":") && allow_tensor)
                {
                    string op_name = name.Split(':')[0];
                    int out_n = int.Parse(name.Split(':')[1]);

                    if (_nodes_by_name.ContainsKey(op_name))
                        return _nodes_by_name[op_name].outputs[out_n];
                }
                else if(!name.Contains(":") & allow_operation)
                {
                    if (!_nodes_by_name.ContainsKey(name))
                        throw new KeyError($"The name {name} refers to an Operation not in the graph.");
                    return _nodes_by_name[name];
                }
                else if (!name.Contains(":") & !allow_operation)
                {
                    throw new NotImplementedException("_as_graph_element_locked");
                }
            }

            if (obj is Tensor tensor && allow_tensor)
            {
                if (tensor.Graph.Equals(this))
                {
                    return tensor;
                }
                else
                {
                    throw new Exception($"Tensor {obj} is not an element of this graph.");
                }
            }
            else if (obj is Operation op && allow_operation)
            {
                if (op.graph.Equals(this))
                {
                    return op;
                }
                else
                {
                    throw new Exception($"Operation {obj} is not an element of this graph.");
                }
            }

            throw new Exception($"Can not convert a {obj.GetType().Name} into a {types_str}.");
        }

        public void add_to_collection<T>(string name, T value)
        {
            _check_not_finalized();
            if (_collections.ContainsKey(name))
                (_collections[name] as List<T>).Add(value);
            else
                _collections[name] = new List<T> { value };
        }

        public void add_to_collections<T>(List<string> names, T value)
        {
            foreach (string name in names)
                add_to_collection(name, value);
        }

        private void _check_not_finalized()
        {
            if (_finalized)
                throw new RuntimeError("Graph is finalized and cannot be modified.");
        }

        public unsafe Operation create_op(string op_type, Tensor[] inputs, TF_DataType[] dtypes, 
            TF_DataType[] input_types = null, string name = "", 
            Dictionary<string, AttrValue> attrs = null, OpDef op_def = null)
        {
            if (inputs == null)
                inputs = new Tensor[0];

            foreach ((int idx, Tensor a) in Python.enumerate(inputs))
            {

            }

            if (String.IsNullOrEmpty(name))
                name = op_type;
            // If a names ends with a '/' it is a "name scope" and we use it as-is,
            // after removing the trailing '/'.
            name = name.EndsWith("/") ? ops._name_from_scope_name(name) : unique_name(name);
            var node_def = ops._NodeDef(op_type, name, device: "", attrs: attrs);

            var input_ops = inputs.Select(x => x.op).ToArray();
            var control_inputs = _control_dependencies_for_inputs(input_ops);

            var op = new Operation(node_def, 
                this,
                inputs: inputs,
                output_types: dtypes,
                control_inputs: control_inputs,
                input_types: input_types,
                original_op: null,
                op_def: op_def);

            _create_op_helper(op, true);

            Console.Write($"create_op: {op_type} '{node_def.Name}'");
            Console.Write($", inputs: {(inputs.Length == 0 ? "empty" : String.Join(", ", inputs.Select(x => x.name)))}");
            Console.Write($", control_inputs: {(control_inputs.Length == 0 ? "empty" : String.Join(", ", control_inputs.Select(x => x.name)))}");
            Console.Write($", outputs: {(op.outputs.Length == 0 ? "empty" : String.Join(", ", op.outputs.Select(x => x.name)))}");
            Console.WriteLine();

            return op;
        }

        private void _create_op_helper(Operation op, bool compute_device = true)
        {
            _record_op_seen_by_control_dependencies(op);
        }

        public void _add_op(Operation op)
        {
            op._id_value = _next_id();
            _nodes_by_id[op._id] = op;
            _nodes_by_name[op.name] = op;
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
                return !_unfetchable_ops.Contains((tensor_or_op as Operation).name);
            }

            return false;
        }

        public string get_name_scope()
        {
            return _name_stack;
        }

        public string name_scope(string name)
        {
            string new_stack = "";

            if (string.IsNullOrEmpty(name))
                new_stack = "";
            else if (name.EndsWith("/"))
                new_stack = ops._name_from_scope_name(name);
            else
                new_stack = unique_name(name);

            _name_stack = new_stack;

            return String.IsNullOrEmpty(new_stack) ? "" : new_stack + "/";
        }

        public string unique_name(string name, bool mark_as_used = true)
        {
            if (!String.IsNullOrEmpty(_name_stack))
            {
                name = _name_stack + "/" + name;
            }

            var name_key = name.ToLower();
            int i = 0;
            if (_names_in_use.ContainsKey(name_key))
            {
                foreach (var item in _names_in_use)
                {
                    if (item.Key == name_key)
                    {
                        i = _names_in_use[name_key];
                        break;
                    }
                    
                    i++;
                }
            }

            if (mark_as_used)
                if (_names_in_use.ContainsKey(name_key))
                    _names_in_use[name_key]++;
                else
                    _names_in_use[name_key] = i + 1;
            
            if (i > 0)
            {
                var base_name_key = name_key;

                // Make sure the composed name key is not already used.
                if (_names_in_use.ContainsKey(name_key))
                {
                    name_key = $"{base_name_key}_{i}";
                    i += 1;
                }

                if (mark_as_used)
                    _names_in_use[name_key] = 1;

                name = $"{name}_{i - 1}";
            }

            return name;
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

        public ITensorOrOperation[] get_operations()
        {
            return _nodes_by_name.Values.Select(x => x).ToArray();
        }

        public string[] get_all_collection_keys()
        {
            return _collections.Keys.Where(x => !x.StartsWith("__")).ToArray();
        }

        public object get_collection(string name, string scope = "")
        {
            return _collections.ContainsKey(name) ? _collections[name] : null;
        }

        public object get_collection_ref(string name)
        {
            if (!_collections.ContainsKey(name))
                _collections[name] = new List<object>();
            return _collections[name];
        }

        public void Dispose()
        {
            c_api.TF_DeleteGraph(_handle);
        }

        public void __enter__()
        {
        }

        public void __exit__()
        {
            
        }

        public static implicit operator IntPtr(Graph graph)
        {
            return graph._handle;
        }
    }
}
