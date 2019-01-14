using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public class Operation
    {
        private readonly IntPtr _handle;

        public Graph Graph { get; }
        public int _id => _id_value;
        private int _id_value;

        private Status status = new Status();

        public string Name => c_api.StringPiece(c_api.TF_OperationName(_handle));
        public string OpType => c_api.StringPiece(c_api.TF_OperationOpType(_handle));
        public string Device => c_api.StringPiece(c_api.TF_OperationDevice(_handle));

        public int NumOutputs => c_api.TF_OperationNumOutputs(_handle);
        public TF_DataType OutputType(int index) => c_api.TF_OperationOutputType(new TF_Output(_handle, index));
        public int OutputListLength(string name) => c_api.TF_OperationOutputListLength(_handle, name, status);

        public TF_Output Input(int index) => c_api.TF_OperationInput(new TF_Input(_handle, index));
        public TF_DataType InputType(int index) => c_api.TF_OperationInputType(new TF_Input(_handle, index));
        public int InputListLength(string name) => c_api.TF_OperationInputListLength(_handle, name, status);
        public int NumInputs => c_api.TF_OperationNumInputs(_handle);

        public int OutputNumConsumers(int index) => c_api.TF_OperationOutputNumConsumers(new TF_Output(_handle, index));
        public unsafe TF_Input[] OutputConsumers(int index, int max_consumers)
        {
            int size = Marshal.SizeOf<TF_Input>();
            var handle = Marshal.AllocHGlobal(size);
            int num = c_api.TF_OperationOutputConsumers(new TF_Output(_handle, index), handle, max_consumers);
            var consumers = new TF_Input[num];
            for(int i = 0; i < num; i++)
            {
                consumers[i] = Marshal.PtrToStructure<TF_Input>(handle + i * size);
            }

            return consumers;
        }

        public int NumControlInputs => c_api.TF_OperationNumControlInputs(_handle);

        public unsafe Operation[] GetControlInputs()
        {
            var control_inputs = new Operation[NumControlInputs];

            if(NumControlInputs > 0)
            {
                IntPtr control_input_handle = Marshal.AllocHGlobal(Marshal.SizeOf<IntPtr>() * NumControlInputs);
                c_api.TF_OperationGetControlInputs(_handle, control_input_handle, NumControlInputs);
                for (int i = 0; i < NumControlInputs; i++)
                {
                    var handle = control_input_handle + Marshal.SizeOf<IntPtr>() * i;
                    control_inputs[i] = new Operation(*(IntPtr*)handle);
                }
            }

            return control_inputs;
        }

        public int NumControlOutputs => c_api.TF_OperationNumControlOutputs(_handle);

        public unsafe Operation[] GetControlOutputs()
        {
            var control_outputs = new Operation[NumControlOutputs];

            if(NumControlOutputs > 0)
            {
                IntPtr control_output_handle = Marshal.AllocHGlobal(Marshal.SizeOf<IntPtr>() * NumControlOutputs);
                c_api.TF_OperationGetControlOutputs(_handle, control_output_handle, NumControlInputs);
                for (int i = 0; i < NumControlInputs; i++)
                {
                    var handle = control_output_handle + Marshal.SizeOf<IntPtr>() * i;
                    control_outputs[i] = new Operation(*(IntPtr*)handle);
                }
            }

            return control_outputs;
        }

        private Tensor[] _outputs;
        public Tensor[] outputs => _outputs;
        public Tensor[] inputs;

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

        public Operation(NodeDef node_def, Graph g, List<Tensor> inputs = null, TF_DataType[] output_types = null, object control_inputs = null, TF_DataType[] input_types = null, string original_op = "", OpDef op_def = null)
        {
            Graph = g;

            _id_value = Graph._next_id();
            if(op_def == null)
                op_def = g.GetOpDef(node_def.Op);

            _handle = ops._create_c_op(g, node_def, inputs);
            
            _outputs = new Tensor[NumOutputs];
            output_types = new TF_DataType[NumOutputs];

            for (int i = 0; i < NumOutputs; i++)
            {
                output_types[i] = OutputType(i);
                _outputs[i] = new Tensor(this, i, output_types[i]);
            }

            Graph._add_op(this);
        }

        public object get_attr(string name)
        {
            object ret = null;

            var fields = new string[] { "s", "i", "f", "b", "type", "shape", "tensor", "func" };

            switch (name)
            {
                case "dtype":
                    ret = _outputs[0];
                    break;
                case "shape":
                    ret = new TensorShapeProto();
                    break;
            }

            return ret;
        }

        public TF_AttrMetadata GetAttributeMetadata(string attr_name, Status s)
        {
            return c_api.TF_OperationGetAttrMetadata(_handle, attr_name, s);
        }

        public NodeDef GetNodeDef()
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
