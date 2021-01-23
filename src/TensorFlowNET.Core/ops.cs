/*****************************************************************************
   Copyright 2018 The TensorFlow.NET Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

using Google.Protobuf;
using Google.Protobuf.Collections;
using NumSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using Tensorflow.Contexts;
using Tensorflow.Eager;
using Tensorflow.Graphs;
using Tensorflow.Util;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public partial class ops
    {
        public static long tensor_id(Tensor tensor)
        {
            return tensor.Id;
        }

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
        public static object get_collection(string key, string scope = null)
        {
            return get_default_graph().get_collection(key, scope);
        }

        public static List<T> get_collection<T>(string key, string scope = null)
        {
            return get_default_graph().get_collection<T>(key, scope);
        }

        public static List<T> get_collection_ref<T>(string key)
        {
            return get_default_graph().get_collection_ref<T>(key);
        }

        public static Graph _get_graph_from_inputs(params object[] op_input_list)
        {
            var current_default_graph = get_default_graph();
            if (current_default_graph.building_function)
                return current_default_graph;

            Graph graph = null;
            foreach (var op_input in op_input_list)
            {
                if (op_input is Tensor op_input_tensor)
                    graph = graph ?? op_input_tensor.graph;
            }
            return graph ?? current_default_graph;
        }

        public static Graph _get_graph_from_inputs(Tensors op_input_list)
            => _get_graph_from_inputs(op_input_list: op_input_list, graph: null);

        public static Graph _get_graph_from_inputs(Tensors op_input_list, Graph graph = null)
        {
            foreach (var op_input in op_input_list)
            {
                // Determine if this is a valid graph_element.
                // var graph_element = op_input;
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
        public static Tensor convert_to_tensor(object value,
            TF_DataType dtype = TF_DataType.DtInvalid,
            string name = null,
            bool as_ref = false,
            TF_DataType preferred_dtype = TF_DataType.DtInvalid,
            Context ctx = null)
        {
            if (dtype == TF_DataType.DtInvalid)
                dtype = preferred_dtype;

            if (value is EagerTensor eager_tensor)
            {
                if (tf.executing_eagerly())
                    return eager_tensor;
                else
                {
                    var graph = get_default_graph();
                    if (!graph.building_function)
                        throw new RuntimeError("Attempting to capture an EagerTensor without building a function.");
                    return (graph as FuncGraph).capture(eager_tensor, name: name);
                }
            }

            Tensor ret = value switch
            {
                NDArray nd => constant_op.constant(nd, dtype: dtype, name: name),
                EagerTensor tensor => tensor.dtype == TF_DataType.TF_RESOURCE
                            ? tensor.AsPlaceholder(name: name)
                            : tensor.AsConstant(name: name),
                Tensor tensor => tensor,
                Tensor[] tensors => array_ops._autopacking_helper(tensors, dtype, name == null ? "packed" : name),
                RefVariable varVal => varVal._TensorConversionFunction(dtype: dtype, name: name, as_ref: as_ref),
                ResourceVariable varVal => varVal._TensorConversionFunction(dtype: dtype, name: name, as_ref: as_ref),
                TensorShape ts => constant_op.constant(ts.dims, dtype: dtype, name: name),
                int[] dims => constant_op.constant(dims, dtype: dtype, name: name),
                string str => constant_op.constant(str, dtype: tf.@string, name: name),
                string[] str => constant_op.constant(str, dtype: tf.@string, name: name),
                IEnumerable<object> objects => array_ops._autopacking_conversion_function(objects, dtype: dtype, name: name),
                _ => constant_op.constant(value, dtype: dtype, name: name)
            };

            return ret;
        }


        public static Tensor convert_to_tensor_or_composite(Tensor value, TF_DataType dtype = TF_DataType.DtInvalid, string name = null)
        {
            return internal_convert_to_tensor_or_composite(value: value, dtype: dtype, name: name, as_ref: false);
        }

        public static Tensor internal_convert_to_tensor_or_composite(Tensor value, TF_DataType dtype = TF_DataType.DtInvalid, string name = null, bool as_ref = false)
            => convert_to_tensor(value, dtype: dtype, name: name, as_ref: as_ref);

        /// <summary>
        /// Wrapper for `Graph.control_dependencies()` using the default graph.
        /// 
        /// See `tf.Graph.control_dependencies` for more details.
        ///
        /// When eager execution is enabled, any callable object in the `control_inputs`
        /// list will be called.
        /// </summary>
        /// <param name="control_inputs">
        /// A list of `Operation` or `Tensor` objects which
        /// must be executed or computed before running the operations
        /// defined in the context.Can also be `None` to clear the control
        /// dependencies.If eager execution is enabled, any callable object in the
        /// `control_inputs` list will be called.
        /// </param>
        /// <returns>
        /// A context manager that specifies control dependencies for all
        /// operations constructed within the context.
        /// </returns>
        public static _ControlDependenciesController control_dependencies(object[] control_inputs)
            => get_default_graph().control_dependencies(control_inputs);

        /// <summary>
        /// Creates a TF_Operation.
        /// </summary>
        /// <param name="graph">a `Graph`.</param>
        /// <param name="node_def">`node_def_pb2.NodeDef` for the operation to create.</param>
        /// <param name="inputs">
        /// A list of `Tensor`s (corresponding to scalar inputs) and lists of
        /// `Tensor`s (corresponding to sequence inputs, e.g. "int64 * N",
        /// "list(int64)"). The length of the list should be equal to the number of
        /// inputs specified by this operation's op def.
        /// </param>
        /// <param name="control_inputs">A list of `Operation`s to set as control dependencies.</param>
        /// <returns>A wrapped TF_Operation*.</returns>
        public static (IntPtr, OperationDescription) _create_c_op(Graph graph, NodeDef node_def, Tensor[] inputs, Operation[] control_inputs,
            OpDef op_def = null)
        {
            if (op_def == null)
                op_def = graph.GetOpDef(node_def.Op);

            var input_tensors = _reconstruct_sequence_inputs(op_def, inputs, node_def.Attr);

            lock (Locks.ProcessWide)
            {
                var op_desc = graph.NewOperation(node_def.Op, node_def.Name);

                if (!string.IsNullOrEmpty(node_def.Device))
                    c_api.TF_SetDevice(op_desc, node_def.Device);

                // Add inputs
                foreach (var op_input in input_tensors)
                {
                    if (op_input.IsList)
                        c_api.TF_AddInputList(op_desc, op_input.Select(x => x._as_tf_output()).ToArray(), op_input.Count());
                    else if (op_input.Count() == 1)
                        c_api.TF_AddInput(op_desc, op_input[0]._as_tf_output());
                }

                var status = tf.Status;

                // Add control inputs
                foreach (var control_input in control_inputs)
                    c_api.TF_AddControlInput(op_desc, control_input);

                // Add attrs
                foreach (var attr in node_def.Attr)
                {
                    var bytes = attr.Value.ToByteArray();
                    c_api.TF_SetAttrValueProto(op_desc, attr.Key, bytes, proto_len: bytes.Length, status: status.Handle);
                    status.Check(true);
                }

                var c_op = c_api.TF_FinishOperation(op_desc, status.Handle);

                status.Check(true);

                return (c_op, op_desc);
            }
        }

        public static Tensors[] _reconstruct_sequence_inputs(OpDef op_def, Tensor[] inputs, MapField<string, AttrValue> attrs)
        {
            var grouped_inputs = new List<Tensors>();
            int i = 0;

            foreach (var input_arg in op_def.InputArg)
            {
                int input_len = 1;
                bool is_sequence = false;

                if (!string.IsNullOrEmpty(input_arg.NumberAttr))
                {
                    input_len = (int)attrs[input_arg.NumberAttr].I;
                    is_sequence = true;
                }
                else if (!string.IsNullOrEmpty(input_arg.TypeListAttr))
                {
                    input_len = attrs[input_arg.TypeListAttr].List.Type.Count;
                    is_sequence = true;
                }

                if (is_sequence)
                {
                    var input_tensors = new Tensors(inputs.Skip(i).Take(input_len).ToArray());
                    input_tensors.IsList = true;
                    grouped_inputs.Add(input_tensors);
                }
                else
                    grouped_inputs.Add(inputs[i]);

                i += input_len;
            }

            return grouped_inputs.ToArray();
        }

        public static OpDef _get_op_def(Graph graph, string type)
        {
            return graph.GetOpDef(type);
        }

        public static NodeDef _NodeDef(string op_type, string name, Dictionary<string, AttrValue> attrs = null)
        {
            var node_def = new NodeDef();
            node_def.Op = op_type;
            node_def.Name = name;

            if (attrs != null)
            {
                foreach (var attr in attrs)
                    node_def.Attr.Add(attr.Key, attr.Value);
            }

            return node_def;
        }

        public static string name_from_scope_name(string name)
        {
            if (name == null)
                return null;
            else if (name.EndsWith("/"))
                return name.Substring(0, name.Length - 1);
            else
                return name;
        }

        /// <summary>
        /// A context manager that lifts ops out of control-flow scopes and function-building graphs.
        /// </summary>
        /// <returns></returns>
        public static NameScope init_scope()
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

            tf_with(ops.control_dependencies(null), delegate
            {
                // var outer_graph = get_default_graph();
                // outer_device_stack = None
            });

            tf.Context.ScopeName = scope;
            return ops.name_scope(scope);
        }

        private static int uid_number = -1;

        /// <summary>
        /// A unique (within this program execution) integer.
        /// Not thread safe
        /// </summary>
        /// <returns></returns>
        public static int uid()
        {
            return Interlocked.Increment(ref uid_number);
        }

        public static void reset_uid()
        {
            uid_number = -1;
        }

        public static void colocate_with(bool ignore_existing = false)
        {
            _colocate_with_for_gradient(null, null, ignore_existing);
        }

        public static void colocate_with(Operation op, bool ignore_existing = false)
        {
            _colocate_with_for_gradient(op, null, ignore_existing);
        }

        public static void colocate_with(Tensor tensor, bool ignore_existing = false)
        {
            _colocate_with_for_gradient(tensor.op, null, ignore_existing);
        }

        public static void colocate_with(IVariableV1 variable, bool ignore_existing = false)
        {
            _colocate_with_for_gradient(variable.AsTensor(), null, ignore_existing);
        }

        public static void _colocate_with_for_gradient(Operation op, string gradient_uid, bool ignore_existing = false)
        {
            var default_graph = get_default_graph();
            default_graph._colocate_with_for_gradient(op, gradient_uid, ignore_existing);
        }

        /// <summary>
        /// Uses the default session to evaluate one or more tensors.
        /// </summary>
        /// <param name="tensor">A single Tensor, or a list of Tensor objects.</param>
        /// <param name="feed_dict">
        /// A dictionary that maps Tensor objects (or tensor names) to lists,
        /// numpy ndarrays, TensorProtos, or strings.
        /// </param>
        /// <param name="graph">The graph in which the tensors are defined.</param>
        /// <param name="session">A different session to use to evaluate "tensors".</param>
        /// <returns>
        /// Either a single numpy ndarray if "tensors" is a single tensor; or a list
        /// of numpy ndarrays that each correspond to the respective element in
        /// "tensors".
        /// </returns>
        public static NDArray _eval_using_default_session(Tensor tensor, FeedItem[] feed_dict, Graph graph, Session session = null)
        {
            if (session == null)
            {
                session = get_default_session();

                if (session == null)
                    throw new ValueError("Cannot evaluate tensor using `eval()`: No default " +
                           "session is registered. Use `with " +
                           "sess.as_default()` or pass an explicit session to " +
                           "`eval(session=sess)`");

                if (session.graph != graph)
                    throw new ValueError("Cannot use the default session to evaluate tensor: " +
                           "the tensor's graph is different from the session's " +
                           "graph. Pass an explicit session to " +
                           "`eval(session=sess)`.");
            }
            else
            {
                if (session.graph != graph)
                    throw new ValueError("Cannot use the default session to evaluate tensor: " +
                           "the tensor's graph is different from the session's " +
                           "graph. Pass an explicit session to " +
                           "`eval(session=sess)`.");
            }

            return session.run(tensor, feed_dict);
        }

        /// <summary>
        /// Prepends name scope to a name.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="import_scope"></param>
        /// <returns></returns>
        public static string prepend_name_scope(string name, string import_scope)
        {
            if (!string.IsNullOrEmpty(import_scope))
            {
                if (import_scope.EndsWith("/"))
                    import_scope = import_scope.Substring(0, import_scope.Length - 1);

                return $"{import_scope}/{name}";
            }
            else
                return name;
        }

        public static void _run_using_default_session(Operation operation, FeedItem[] feed_dict, Graph graph, Session session)
        {
            if (session == null)
            {
                session = get_default_session();
                if (session == null)
                    throw new ValueError("Cannot execute operation using `run()`: No default " +
                       "session is registered. Use `with " +
                       "sess.as_default():` or pass an explicit session to " +
                       "`run(session=sess)`");
            }

            if (session.graph != graph)
                throw new ValueError("Cannot use the default session to execute operation: " +
                   "the operation's graph is different from the " +
                   "session's graph. Pass an explicit session to " +
                   "run(session=sess).");

            session.run(operation, feed_dict);
        }

        public static Tensor[] convert_n_to_tensor(object[] values, TF_DataType dtype = TF_DataType.DtInvalid, string name = null)
            => internal_convert_n_to_tensor(values, dtype: dtype, name: name, as_ref: false);

        public static Tensor[] convert_n_to_tensor_or_indexed_slices(Tensor[] values, TF_DataType dtype = TF_DataType.DtInvalid, string name = null)
            => internal_convert_n_to_tensor_or_indexed_slices(values, dtype: dtype, name: name);

        public static Tensor convert_to_tensor_or_indexed_slices(Tensor value, TF_DataType dtype = TF_DataType.DtInvalid, string name = null)
            => internal_convert_to_tensor_or_indexed_slices(value: value, dtype: dtype, name: name, as_ref: false);

        public static Tensor internal_convert_to_tensor_or_indexed_slices(Tensor value, TF_DataType dtype = TF_DataType.DtInvalid, string name = null, bool as_ref = false)
            => value;

        public static Tensor[] internal_convert_n_to_tensor_or_indexed_slices(Tensor[] values, TF_DataType dtype = TF_DataType.DtInvalid, string name = null, bool as_ref = false)
        {
            var ret = new List<Tensor>();

            foreach (var (i, value) in enumerate(values))
            {
                if (value == null)
                {
                    ret.Add(value);
                }
                else
                {
                    var n = string.IsNullOrEmpty(name) ? "" : $"{name}_{i}";
                    ret.Add(internal_convert_to_tensor_or_indexed_slices(value, dtype: dtype, name: n, as_ref: as_ref));
                }
            }

            return ret.ToArray();
        }

        public static Tensor[] internal_convert_n_to_tensor(object[] values, TF_DataType dtype = TF_DataType.DtInvalid,
            string name = null, TF_DataType preferred_dtype = TF_DataType.DtInvalid,
            bool as_ref = false)
        {
            var ret = new List<Tensor>();
            foreach ((int i, object value) in enumerate(values))
            {
                string n = string.IsNullOrEmpty(name) ? "" : $"{name}_{i}";
                ret.Add(convert_to_tensor(value, dtype: dtype, name: n, as_ref: as_ref, preferred_dtype: preferred_dtype));
            }
            return ret.ToArray();
        }

        public static string strip_name_scope(string name, string export_scope = "")
        {
            if (!string.IsNullOrEmpty(export_scope))
            {
                throw new NotImplementedException("ops.strip_name_scope");
            }
            else
            {
                return name;
            }
        }

        public static string get_name_scope()
        {
            var g = get_default_graph();
            return g.get_name_scope();
        }

        public static bool executing_eagerly_outside_functions()
        {
            if (tf.Context.executing_eagerly())
                return true;
            else
                throw new NotImplementedException("");
        }
    }
}
