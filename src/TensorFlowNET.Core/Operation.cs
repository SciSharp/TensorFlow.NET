using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;
using TF_DataType = Tensorflow.DataType;

namespace TensorFlowNET.Core
{
    public class Operation
    {
        private Graph _graph;
        private IntPtr _c_op;
        public int _id => _id_value;
        private int _id_value;
        public string name;
        private Tensor[] _outputs;
        public Tensor[] outputs => _outputs;

        public Operation(NodeDef node_def, Graph g, object inputs = null, TF_DataType[] output_types = null, object control_inputs = null, TF_DataType[] input_types = null, string original_op = "", string op_def = "")
        {
            _graph = g;

            _id_value = _graph._next_id();
            _c_op = ops._create_c_op(g, node_def, inputs);
            var num_outputs = c_api.TF_OperationNumOutputs(_c_op);

            _outputs = new Tensor[num_outputs];
            for (int i = 0; i < num_outputs; i++)
            {
                _outputs[i] = new Tensor(this, i, TF_DataType.DtDouble);
            }

            _graph._add_op(this);
        }
    }
}
