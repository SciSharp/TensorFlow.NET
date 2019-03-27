using Google.Protobuf.Collections;
//using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public partial class Operation : ITensorOrOperation
    {
        private readonly IntPtr _handle; // _c_op in python

        private Graph _graph;
        //[JsonIgnore]
        public Graph graph => _graph;
        //[JsonIgnore]
        public int _id => _id_value;
        //[JsonIgnore]
        public int _id_value;

        public string type => OpType;
        //[JsonIgnore]
        public Operation op => this;
        public TF_DataType dtype => TF_DataType.DtInvalid;
        private Status status = new Status();

        public string name => c_api.StringPiece(c_api.TF_OperationName(_handle));
        public string OpType => c_api.StringPiece(c_api.TF_OperationOpType(_handle));
        public string Device => c_api.StringPiece(c_api.TF_OperationDevice(_handle));

        private NodeDef _node_def;
        public NodeDef node_def
        {
            get
            {
                if(_node_def == null)
                    _node_def = GetNodeDef();

                return _node_def;
            }
        }

        public Operation(IntPtr handle)
        {
            if (handle == IntPtr.Zero)
                return;

            _handle = handle;
            _graph = ops.get_default_graph();
            _outputs = new Tensor[NumOutputs];
            for (int i = 0; i < NumOutputs; i++)
                _outputs[i] = new Tensor(this, i, OutputType(i));
        }

        public Operation(Graph g, string opType, string oper_name)
        {
            _graph = g;

            var desc = c_api.TF_NewOperation(g, opType, oper_name);
            c_api.TF_SetAttrType(desc, "dtype", TF_DataType.TF_INT32);
            c_api.TF_FinishOperation(desc, status);
        }

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
            if(control_inputs != null)
            {
                foreach(var c in control_inputs)
                {
                    switch (c)
                    {
                        case Operation c1:
                            control_input_ops.Add(c1);
                            break;
                        default:
                            throw new NotImplementedException($"Control input must be an Operation, a Tensor, or IndexedSlices: {c}");
                    }
                }
            }

            // Dict mapping op name to file and line information for op colocation
            // context managers.
            _control_flow_context = graph._get_control_flow_context();

            // This will be set by self.inputs.
            if (op_def == null)
                op_def = g.GetOpDef(node_def.Op);

            var grouped_inputs = _reconstruct_sequence_inputs(op_def, inputs, node_def.Attr);
            _handle = ops._create_c_op(g, node_def, grouped_inputs, control_input_ops.ToArray());

            // Initialize self._outputs.
            output_types = new TF_DataType[NumOutputs];
            for (int i = 0; i < NumOutputs; i++)
                output_types[i] = OutputType(i);

            _outputs = new Tensor[NumOutputs];
            for (int i = 0; i < NumOutputs; i++)
                _outputs[i] = new Tensor(this, i, OutputType(i));

            graph._add_op(this);

            if (_handle != IntPtr.Zero)
                _control_flow_post_processing();
        }

        public void run(FeedItem[] feed_dict = null, Session session = null)
        {
            ops._run_using_default_session(this, feed_dict, graph, session);
        }

        private object[] _reconstruct_sequence_inputs(OpDef op_def, Tensor[] inputs, MapField<string, AttrValue> attrs)
        {
            var grouped_inputs = new List<object>();
            int i = 0;
            int input_len = 0;
            bool is_sequence = false;
            foreach (var input_arg in op_def.InputArg)
            {
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
                else
                {
                    input_len = 1;
                    is_sequence = false;
                }

                if (is_sequence)
                    grouped_inputs.Add(inputs.Skip(i).Take(input_len).ToArray());
                else
                    grouped_inputs.Add(inputs[i]);

                i += input_len;
            }

            return grouped_inputs.ToArray();
        }

        public object get_attr(string name)
        {
            AttrValue x = null;

            using (var buf = new Buffer())
            {
                c_api.TF_OperationGetAttrValueProto(_handle, name, buf, status);
                status.Check(true);
                x = AttrValue.Parser.ParseFrom(buf);
            }

            string oneof_value = x.ValueCase.ToString();
            if (string.IsNullOrEmpty(oneof_value))
                return null;

            if(oneof_value == "list")
                throw new NotImplementedException($"Unsupported field type in {x.ToString()}");

            if (oneof_value == "type")
                return x.Type;

            object result = x.GetType().GetProperty(oneof_value).GetValue(x);
            if (result is Google.Protobuf.ByteString byteString)
                return byteString.ToStringUtf8();
            return result;
        }

        public TF_AttrMetadata GetAttributeMetadata(string attr_name, Status s)
        {
            return c_api.TF_OperationGetAttrMetadata(_handle, attr_name, s);
        }

        private NodeDef GetNodeDef()
        {
            using (var s = new Status())
            using (var buffer = new Buffer())
            {
                c_api.TF_OperationToNodeDef(_handle, buffer, s);
                s.Check();
                return NodeDef.Parser.ParseFrom(buffer);
            }
        }

        public override string ToString()
        {
            return _handle == IntPtr.Zero ? "tf.Operation Undefined" : $"tf.Operation '{name}' type={OpType}";
        }

        public static implicit operator Operation(IntPtr handle) => new Operation(handle);
        public static implicit operator IntPtr(Operation op) => op._handle;

        public override bool Equals(object obj)
        {
            switch (obj)
            {
                case IntPtr val:
                    return val == _handle;
                case Operation val:
                    return val._handle == _handle;
            }

            return base.Equals(obj);
        }
    }
}
