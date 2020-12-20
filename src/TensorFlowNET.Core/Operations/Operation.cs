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

using NumSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Util;
using static Tensorflow.Binding;

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
        private readonly IntPtr _handle; // _c_op in python

        private readonly Graph _graph;
        private NodeDef _node_def;

        public string type => OpType;

        public Graph graph => _graph;

        public int _id => _id_value;

        public int _id_value { get; set; }
        public Operation op => this;
        public TF_DataType dtype => TF_DataType.DtInvalid;
        public virtual string name => _handle == IntPtr.Zero ? null : c_api.StringPiece(c_api.TF_OperationName(_handle));
        public string OpType => _handle == IntPtr.Zero ? null : c_api.StringPiece(c_api.TF_OperationOpType(_handle));

        public string Device => _handle == IntPtr.Zero ? null : c_api.StringPiece(c_api.TF_OperationDevice(_handle));

        bool _is_stateful;
        public OperationDescription OpDesc { get; set; }

        public NodeDef node_def
        {
            get
            {
                if (_node_def == null)
                    _node_def = GetNodeDef();

                return _node_def;
            }
        }

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

            (_handle, OpDesc) = ops._create_c_op(g, node_def, inputs, control_input_ops.ToArray(), op_def);
            _is_stateful = op_def.IsStateful;

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
            => (T)get_attr(name);

        public virtual T[] get_attr_list<T>(string name)
        {
            if (tf.executing_eagerly())
                return (T[])get_attr(name);

            AttrValue x = null;

            lock (Locks.ProcessWide)
            {
                using var buf = new Buffer();
                c_api.TF_OperationGetAttrValueProto(_handle, name, buf.Handle, tf.Status.Handle);
                tf.Status.Check(true);

                x = AttrValue.Parser.ParseFrom(buf.DangerousMemoryBlock.Stream());
            }

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
            AttrValue x = null;

            lock (Locks.ProcessWide)
            {
                using var buf = new Buffer();
                c_api.TF_OperationGetAttrValueProto(_handle, name, buf.Handle, tf.Status.Handle);
                tf.Status.Check(true);

                x = AttrValue.Parser.ParseFrom(buf.DangerousMemoryBlock.Stream());
            }

            string oneof_value = x.ValueCase.ToString();
            if (string.IsNullOrEmpty(oneof_value))
                return null;

            switch (oneof_value.ToLower())
            {
                case "list":
                    throw new NotImplementedException($"Unsupported field type in {oneof_value}");
                case "type":
                    return x.Type;
                case "s":
                    return x.S.ToStringUtf8();
                default:
                    return x.GetType().GetProperty(oneof_value).GetValue(x);
            }
        }

        public TF_AttrMetadata GetAttributeMetadata(string attr_name, Status s)
        {
            return c_api.TF_OperationGetAttrMetadata(_handle, attr_name, s.Handle);
        }

        private NodeDef GetNodeDef()
        {
            lock (Locks.ProcessWide)
                using (var s = new Status())
                using (var buffer = new Buffer())
                {
                    c_api.TF_OperationToNodeDef(_handle, buffer.Handle, s.Handle);
                    s.Check();

                    return NodeDef.Parser.ParseFrom(buffer.DangerousMemoryBlock.Stream());
                }
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

            var input = _tf_input(index);
            var output = tensor._as_tf_output();

            // Reset cached inputs.
            _inputs_val = null;
            _node_def = null;
            // after the c_api call next time _inputs is accessed 
            // the updated inputs are reloaded from the c_api
            lock (Locks.ProcessWide)
            {
                // disable
                // c_api.TF_UpdateEdge(_graph, output, input, tf.Status.Handle);
                //var updated_inputs = inputs;
                tf.Status.Check();
            }
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
            return new TF_Output(op, output_idx);
        }

        /// <summary>
        /// Create and return a new TF_Input for input_idx'th input of this op.
        /// </summary>
        public TF_Input _tf_input(int input_idx)
        {
            return new TF_Input(op, input_idx);
        }

        public NDArray numpy() => throw new NotImplementedException("");
    }
}