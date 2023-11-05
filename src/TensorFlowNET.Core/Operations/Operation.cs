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

using Tensorflow.NumPy;
using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Util;
using static Tensorflow.Binding;
using Google.Protobuf;
using Google.Protobuf.WellKnownTypes;
using System.Diagnostics;

namespace Tensorflow
{
    /// <summary>
    /// Represents a graph node that performs computation on tensors.
    /// 
    /// An `Operation` is a node in a TensorFlow `Graph` that takes zero or
    /// more `Tensor` objects as input, and produces zero or more `Tensor`
    /// objects as output. Objects of type `Operation` are created by
    /// calling an op constructor(such as `tf.matmul`)
    /// or `tf.Graph.create_op`.
    /// 
    /// For example `c = tf.matmul(a, b)` creates an `Operation` of type
    /// "MatMul" that takes tensors `a` and `b` as input, and produces `c`
    /// as output.
    /// 
    /// After the graph has been launched in a session, an `Operation` can
    /// be executed by passing it to
    /// `tf.Session.run`.
    /// `op.run()` is a shortcut for calling `tf.get_default_session().run(op)`.
    /// </summary>
    public partial class Operation : ITensorOrOperation
    {
        protected IntPtr _handle; // _c_op in python

        protected Graph _graph;

        internal Func<Operation, object[], Tensor[]> _gradient_function;

        public string type => OpType;

        public Graph graph => _graph;

        public int _id => _id_value;

        public int _id_value { get; set; }
        public Operation op => this;
        public TF_DataType dtype => output.dtype;
        public virtual string name => _handle == IntPtr.Zero ? "" : c_api.StringPiece(c_api.TF_OperationName(_handle));
        public string OpType => _handle == IntPtr.Zero ? "" : c_api.StringPiece(c_api.TF_OperationOpType(_handle));

        public string Device => _handle == IntPtr.Zero ? "" : c_api.StringPiece(c_api.TF_OperationDevice(_handle));

        //private OperationDescription _op_desc;

        public NodeDef node_def => GetNodeDef();
        protected Operation() { }

        public Operation(IntPtr handle, Graph g = null)
        {
            if (handle == IntPtr.Zero)
                return;

            _handle = handle;
            _graph = g ?? ops.get_default_graph();
            _outputs = new Tensor[NumOutputs];
            for (int i = 0; i < NumOutputs; i++)
                _outputs[i] = new Tensor(this, i, OutputType(i));

            // Dict mapping op name to file and line information for op colocation
            // context managers.
            _control_flow_context = _graph._get_control_flow_context();

            // Note: _control_flow_post_processing() must not be called here, the caller is responsible for calling it when using this constructor.
        }

        /*public Operation(Graph g, string opType, string oper_name)
        {
            _graph = g;

            var _operDesc = c_api.TF_NewOperation(g, opType, oper_name);
            c_api.TF_SetAttrType(_operDesc, "dtype", TF_DataType.TF_INT32);
            lock (Locks.ProcessWide)
                using (var status = new Status())
                {
                    _handle = c_api.TF_FinishOperation(_operDesc, status);
                    status.Check(true);
                }

            // Dict mapping op name to file and line information for op colocation
            // context managers.
            _control_flow_context = graph._get_control_flow_context();
        }*/

        /// <summary>
        /// Creates an `Operation`.
        /// </summary>
        /// <param name="node_def">`node_def_pb2.NodeDef`.  `NodeDef` for the `Operation`.</param>
        /// <param name="g">`Graph`. The parent graph.</param>
        /// <param name="inputs">list of `Tensor` objects. The inputs to this `Operation`.</param>
        /// <param name="output_types">list of `DType` objects.</param>
        /// <param name="control_inputs">
        /// list of operations or tensors from which to have a
        /// control dependency.
        /// </param>
        /// <param name="input_types">
        /// List of `DType` objects representing the
        /// types of the tensors accepted by the `Operation`. By default
        /// uses `[x.dtype.base_dtype for x in inputs]`.  Operations that expect
        /// reference-typed inputs must specify these explicitly.
        /// </param>
        /// <param name="original_op"></param>
        /// <param name="op_def"></param>
        public Operation(NodeDef node_def, Graph g, Tensor[] inputs = null, TF_DataType[] output_types = null, ITensorOrOperation[] control_inputs = null, TF_DataType[] input_types = null, string original_op = "", OpDef op_def = null)
        {
            _graph = g;
            // Build the list of control inputs.
            var control_input_ops = new List<Operation>();
            if (control_inputs != null)
            {
                foreach (var c in control_inputs)
                {
                    switch (c)
                    {
                        case Operation c1:
                            control_input_ops.Add(c1);
                            break;
                        case Tensor tensor:
                            control_input_ops.Add(tensor.op);
                            break;
                        // TODO: IndexedSlices don't yet exist, but once they do, this needs to be uncommented
                        //case IndexedSlices islices:
                        //    control_input_ops.Add(islices.op);
                        //    break;
                        default:
                            throw new NotImplementedException($"Control input must be an Operation, a Tensor, or IndexedSlices: {c}");
                    }
                }
            }

            _id_value = _graph._next_id();

            // Dict mapping op name to file and line information for op colocation
            // context managers.
            _control_flow_context = graph._get_control_flow_context();

            // This will be set by self.inputs.
            if (op_def == null)
                op_def = g.GetOpDef(node_def.Op);

            (_handle, _) = ops._create_c_op(g, node_def, inputs, control_input_ops.ToArray(), op_def);

            // Initialize self._outputs.
            output_types = new TF_DataType[NumOutputs];
            for (int i = 0; i < NumOutputs; i++)
                output_types[i] = OutputType(i);

            _outputs = new Tensor[NumOutputs];
            for (int i = 0; i < NumOutputs; i++)
                _outputs[i] = new Tensor(this, i, output_types[i]);

            graph._add_op(this);

            if (_handle != IntPtr.Zero)
                _control_flow_post_processing();
        }

        public void run(FeedItem[] feed_dict = null, Session session = null)
        {
            ops._run_using_default_session(this, feed_dict, graph, session);
        }

        public virtual T get_attr<T>(string name)
        {
            if (typeof(T).IsValueType)
            {
                return (T)Convert.ChangeType(get_attr(name), typeof(T));
            }
            else
            {
                return (T)get_attr(name);
            }
        }

        internal unsafe TF_DataType _get_attr_type(string name)
        {
            Status status = new();
            TF_DataType result;
            c_api.TF_OperationGetAttrType(_handle, name, new IntPtr(&result), status);
            status.Check(true);
            return result;
        }

        internal unsafe long _get_attr_int(string name)
        {
            long result;
            c_api.TF_OperationGetAttrInt(_handle, name, new IntPtr(&result), tf.Status);
            tf.Status.Check(true);
            return result;
        }

        internal unsafe bool _get_attr_bool(string name)
        {
            Status status = new();
            bool result;
            c_api.TF_OperationGetAttrBool(_handle, name, new IntPtr(&result), status);
            status.Check(true);
            return result;
        }

        public virtual T[] get_attr_list<T>(string name)
        {
            if (tf.executing_eagerly())
                return (T[])get_attr(name);

            var buf = new Buffer();
            c_api.TF_OperationGetAttrValueProto(_handle, name, buf, tf.Status);
            tf.Status.Check(true);

            var x = AttrValue.Parser.ParseFrom(buf.ToArray());

            string oneof_value = x.ValueCase.ToString();
            if (string.IsNullOrEmpty(oneof_value))
                return null;

            switch (typeof(T).Name)
            {
                case nameof(Int32):
                    return x.List.I.Select(x => (T)Convert.ChangeType(x, typeof(T))).ToArray();
                case nameof(Int64):
                    return x.List.I.Select(x => (T)Convert.ChangeType(x, typeof(T))).ToArray();
                default:
                    return null;
            }
        }

        public virtual object get_attr(string name)
        {
            var buf = new Buffer();
            Status status = new();
            c_api.TF_OperationGetAttrValueProto(_handle, name, buf, status);
            status.Check(true);
            var tf_buffer = c_api.TF_GetBuffer(buf);

            var x = AttrValue.Parser.ParseFrom(tf_buffer.AsSpan<byte>());

            var oneof_value = x.ValueCase;
            if (oneof_value == AttrValue.ValueOneofCase.None)
                return new object[0];

            if(oneof_value == AttrValue.ValueOneofCase.List)
            {
                if (x.List.S is not null && x.List.S.Count > 0)
                {
                    return x.List.S.Select(x => x.ToStringUtf8()).ToArray();
                }
                else if (x.List.I is not null && x.List.I.Count > 0)
                {
                    return x.List.I.ToArray();
                }
                else if (x.List.F is not null && x.List.F.Count > 0)
                {
                    return x.List.F.ToArray();
                }
                else if (x.List.B is not null && x.List.B.Count > 0)
                {
                    return x.List.B.ToArray();
                }
                else if (x.List.Shape is not null && x.List.Shape.Count > 0)
                {
                    return x.List.Shape.ToArray();
                }
                else if (x.List.Tensor is not null && x.List.Tensor.Count > 0)
                {
                    return x.List.Tensor.ToArray();
                }
                else if (x.List.Func is not null && x.List.Func.Count > 0)
                {
                    return x.List.Func.ToArray();
                }
                else if (x.List.Type is not null && x.List.Type.Count > 0)
                {
                    return x.List.Type.Select(x => x.as_tf_dtype()).ToArray();
                }
                else
                {
                    return null;
                }
            }
            if(oneof_value == AttrValue.ValueOneofCase.Type)
            {
                return dtypes.as_tf_dtype(x.Type);
            }
            return ProtoUtils.GetSingleAttrValue(x, oneof_value);
        }

        public TF_AttrMetadata GetAttributeMetadata(string attr_name, Status s)
        {
            return c_api.TF_OperationGetAttrMetadata(_handle, attr_name, s);
        }

        [Obsolete("The implementation is not complete.")]
        internal void _set_device_from_string(string device_str)
        {
            // TODO(Rinne): complete it with new C API `SetRequestedDevice`.
            //c_api.TF_SetDevice(_handle, device_str);
        }

        [Obsolete("The implementation is not complete.")]
        internal void _set_device(string device)
        {
            _set_device_from_string(device);
        }

        private NodeDef GetNodeDef()
        {
            var buffer = new Buffer();
            c_api.TF_OperationToNodeDef(_handle, buffer, tf.Status);
            tf.Status.Check(throwException: true);
            return NodeDef.Parser.ParseFrom(buffer.ToArray());
        }

        /// <summary>
        /// Update the input to this operation at the given index.
        /// 
        /// NOTE: This is for TF internal use only.Please don't use it.
        /// </summary>
        /// <param name="index">the index of the input to update.</param>
        /// <param name="tensor"> the Tensor to be used as the input at the given index.</param>
        public void _update_input(int index, Tensor tensor)
        {
            _assert_same_graph(tensor);

            // var input = _tf_input(index);
            // var output = tensor._as_tf_output();

            // Reset cached inputs.
            _inputs_val = null;
            // _node_def = null;
            // after the c_api call next time _inputs is accessed 
            // the updated inputs are reloaded from the c_api
            // lock (Locks.ProcessWide)
            // {
                // disable
                // c_api.TF_UpdateEdge(_graph, output, input, tf.Status.Handle);
                //var updated_inputs = inputs;
                // tf.Status.Check();
            // }
        }

        private void _assert_same_graph(Tensor tensor)
        {
            //TODO: implement
        }

        /// <summary>
        /// Create and return a new TF_Output for output_idx'th output of this op.
        /// </summary>
        public TF_Output _tf_output(int output_idx)
        {
            return new TF_Output(_handle, output_idx);
        }

        /// <summary>
        /// Create and return a new TF_Input for input_idx'th input of this op.
        /// </summary>
        public TF_Input _tf_input(int input_idx)
        {
            return new TF_Input(_handle, input_idx);
        }

        public NDArray numpy() => throw new NotImplementedException("");

        internal void _add_outputs(TF_DataType[] types, Shape[] shapes)
        {
            Debug.Assert(types.Length == shapes.Length);
            int orig_num_outputs = this.outputs.Length;
            var new_outputs = new List<Tensor>(_outputs);

            // Since the `_outputs` is defined as `Array`, when we add new output, we 
            // have to create a new array, which brings some performance concerns.
            // In the future maybe the type of `outputs` should be reconsidered.
            for(int i = 0; i < types.Length; i++)
            {
                var t = new Tensor(this, orig_num_outputs + i, types[i]);
                t.shape = shapes[i];
                new_outputs.Add(t);
            }
            _outputs = new_outputs.ToArray();
        }

        internal void _set_func_attr(string attr_name, string func_name)
        {
            var func = new NameAttrList() { Name = func_name };
            _set_attr(attr_name, new AttrValue() { Func = func });
        }

        internal void _set_type_list_attr(string attr_name, DataType[] types)
        {
            if(types is null || types.Length == 0)
            {
                return;
            }
            var type_list = new AttrValue.Types.ListValue();
            type_list.Type.AddRange(types);
            _set_attr(attr_name, new AttrValue() { List = type_list });
        }

        internal void _set_attr(string attr_name, AttrValue attr_value)
        {
            var buffer = new Buffer(attr_value.ToByteArray());
            try
            {
                _set_attr_with_buf(attr_name, buffer);
            }
            finally
            {
                buffer.Release();
            }
        }

        internal void _set_attr_with_buf(string attr_name, Buffer attr_buf)
        {
            Status status = new();
            c_api.TF_SetAttr(graph, _handle, attr_name, attr_buf, status);
            status.Check(true);
        }
    }
}