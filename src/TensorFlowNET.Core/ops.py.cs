using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using Tensorflow;
using node_def_pb2 = Tensorflow;
using Google.Protobuf;
using System.Linq;

namespace Tensorflow
{
    public partial class ops
    {
        public static void add_to_collection<T>(string name, T value)
        {
            var graph = tf.get_default_graph();
            graph.add_to_collection(name, value);
        }

        public static void add_to_collections<T>(List<string> names, T value)
        {
            var graph = tf.get_default_graph();
            graph.add_to_collections(names, value);
        }

        /// <summary>
        /// Wrapper for `Graph.get_collection()` using the default graph.
        /// contains many standard names for collections.
        /// </summary>
        /// <param name="key">
        /// The key for the collection. For example, the `GraphKeys` class
        /// </param>
        /// <param name="scope"></param>
        /// <returns>
        /// The list of values in the collection with the given `name`, or
        /// an empty list if no value has been added to that collection. The
        /// list contains the values in the order under which they were
        /// collected.
        /// </returns>
        public static object get_collection(string key, string scope = "")
        {
            return get_default_graph().get_collection(key, scope);
        }

        public static Graph get_default_graph()
        {
            return tf.Graph();
        }

        public static Graph _get_graph_from_inputs(List<Tensor> op_input_list, Graph graph = null)
        {
            foreach(var op_input in op_input_list)
            {
                // Determine if this is a valid graph_element.
                var graph_element = op_input;
            }

            return get_default_graph();
        }

        /// <summary>
        /// Converts the given `value` to a `Tensor`.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="dtype"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor convert_to_tensor(object value, TF_DataType dtype = TF_DataType.DtInvalid, string name = "")
        {
            switch (value)
            {
                case Tensor val:
                    return val;
                default:
                    var nd = tensor_util.convert_to_numpy_ndarray(value);
                    return constant_op.Constant(nd, name);
            }
        }

        public static unsafe IntPtr _create_c_op(Graph graph, NodeDef node_def, List<Tensor> inputs)
        {
            var op_desc = graph.NewOperation(node_def.Op, node_def.Name);

            // Add inputs
            if(inputs != null)
            {
                foreach (var op_input in inputs)
                {
                    bool isList = false;
                    if (!isList)
                    {
                        c_api.TF_AddInput(op_desc, op_input._as_tf_output());
                    }
                    else
                    {
                        c_api.TF_AddInputList(op_desc, inputs.Select(x => x._as_tf_output()).ToArray(), inputs.Count);
                    }
                }
            }

            var status = new Status();

            // Add control inputs

            // Add attrs
            foreach (var attr in node_def.Attr)
            {
                var bytes = attr.Value.ToByteArray();
                var proto = Marshal.AllocHGlobal(bytes.Length);
                Marshal.Copy(bytes, 0, proto, bytes.Length);
                
                c_api.TF_SetAttrValueProto(op_desc, attr.Key, proto, proto_len: (uint)bytes.Length, status: status);

                status.Check(true);
            }

            var c_op = c_api.TF_FinishOperation(op_desc, status);

            status.Check(true);

            return c_op;
        }

        public static OpDef _get_op_def(Graph graph, string type)
        {
            return graph.GetOpDef(type);
        }

        public static NodeDef _NodeDef(string op_type, string name, string device = "", Dictionary<string, AttrValue> attrs = null)
        {
            var node_def = new node_def_pb2.NodeDef();
            node_def.Op = op_type;
            node_def.Name = name;

            foreach (var attr in attrs)
            {
                node_def.Attr.Add(attr.Key, attr.Value);
            }
            
            return node_def;
        }

        public static string _name_from_scope_name(string name)
        {
            if (name.EndsWith("/"))
            {
                return name.Substring(0, name.Length - 1);
            }
            else
            {
                return name;
            }
        }

        /// <summary>
        /// A context manager that lifts ops out of control-flow scopes and function-building graphs.
        /// </summary>
        /// <returns></returns>
        public static void init_scope()
        {
            // Retrieve the active name scope: entering an `init_scope` preserves
            // the name scope of the current context.
            var default_graph = get_default_graph();
            var scope = default_graph.get_name_scope();
            if (!String.IsNullOrEmpty(scope) && !scope.EndsWith("/"))
                // Names that end with trailing slashes are treated by `name_scope` as
                // absolute.
                scope += "/";
            // inner_device_stack = default_graph._device_function_stack
            // var outer_context = default_graph.as_default;

            var outer_graph = get_default_graph();
            // outer_device_stack = None
        }

        private static int uid_number = 0;

        /// <summary>
        /// A unique (within this program execution) integer.
        /// Not thread safe
        /// </summary>
        /// <returns></returns>
        public static int uid()
        {
            return uid_number++;
        }

        public static void colocate_with(Operation op, bool ignore_existing = false)
        {
            _colocate_with_for_gradient(op, null, ignore_existing);
        }

        private static void _colocate_with_for_gradient(Operation op, int? gradient_uid, bool ignore_existing = false)
        {
            var default_graph = get_default_graph();
            default_graph._colocate_with_for_gradient(op, gradient_uid, ignore_existing);
        }
    }
}
