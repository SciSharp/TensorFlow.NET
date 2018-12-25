using System;
using System.Collections.Generic;
using System.Text;
using TF_DataType = Tensorflow.DataType;

namespace Tensorflow
{
    public class Operation
    {
        private Graph _graph;
        public Graph graph => _graph;
        public IntPtr _c_op;
        public int _id => _id_value;
        private int _id_value;
        public string name;
        private Tensor[] _outputs;
        public Tensor[] outputs => _outputs;
        public Tensor[] inputs;

        public Operation(Graph g, string opType, string oper_name)
        {
            _graph = g;

            var status = new Status();

            var desc = c_api.TF_NewOperation(g.Handle, opType, oper_name);
            c_api.TF_SetAttrType(desc, "dtype", TF_DataType.TF_INT32);
            c_api.TF_FinishOperation(desc, status.Handle);
        }

        public Operation(NodeDef node_def, Graph g, List<Tensor> inputs = null, TF_DataType[] output_types = null, object control_inputs = null, TF_DataType[] input_types = null, string original_op = "", OpDef op_def = null)
        {
            _graph = g;

            _id_value = _graph._next_id();
            _c_op = ops._create_c_op(g, node_def, inputs);
            var num_outputs = c_api.TF_OperationNumOutputs(_c_op);

            _outputs = new Tensor[num_outputs];
            for (int i = 0; i < num_outputs; i++)
            {
                _outputs[i] = new Tensor(this, i, TF_DataType.TF_FLOAT);
            }

            _graph._add_op(this);
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
    }
}
