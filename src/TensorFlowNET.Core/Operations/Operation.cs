using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public partial class Operation
    {
        private readonly IntPtr _handle;

        public Graph Graph { get; }
        public int _id => _id_value;
        private int _id_value;

        private Status status = new Status();

        public string Name => c_api.StringPiece(c_api.TF_OperationName(_handle));
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
        }

        public Operation(Graph g, string opType, string oper_name)
        {
            Graph = g;

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
        public Operation(NodeDef node_def, Graph g, List<Tensor> inputs = null, TF_DataType[] output_types = null, Operation[] control_inputs = null, TF_DataType[] input_types = null, string original_op = "", OpDef op_def = null)
        {
            Graph = g;

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

            // This will be set by self.inputs.

            _id_value = Graph._next_id();
            if(op_def == null)
                op_def = g.GetOpDef(node_def.Op);

            _handle = ops._create_c_op(g, node_def, inputs, control_input_ops.ToArray());

            output_types = new TF_DataType[NumOutputs];

            for (int i = 0; i < NumOutputs; i++)
                output_types[i] = OutputType(i);

            Graph._add_op(this);
        }

        public object get_attr<T>(string name)
        {
            AttrValue x = null;

            using (var buf = new Buffer())
            {
                c_api.TF_OperationGetAttrValueProto(_handle, name, buf, status);
                status.Check(true);
                x = AttrValue.Parser.ParseFrom(buf);
            }

            switch (name)
            {
                case "T":
                case "dtype":
                    return x.Type;
                case "shape":
                    return x.Shape;
                default:
                    switch (typeof(T).Name)
                    {
                        case "Boolean":
                            return x.B;
                        case "String":
                            return x.S;
                        default:
                            throw new NotImplementedException($"Unsupported field type in {x.ToString()}");
                    }
            }
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
            return _handle == IntPtr.Zero ? "Undefined" : $"'{Name}' type={OpType}";
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
