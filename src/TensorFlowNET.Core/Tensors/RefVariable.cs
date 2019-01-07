using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class RefVariable : Variable
    {
        public bool _in_graph_mode = true;
        public Tensor _initial_value;

        public RefVariable(object initial_value, 
            string name = "",
            TF_DataType trainable = TF_DataType.DtInvalid,
            bool validate_shape = true) : 
            base(initial_value, name, trainable, validate_shape)
        {
            _init_from_args(initial_value, name, trainable);
        }

        private void _init_from_args(object initial_value,
            string name = "",
            TF_DataType trainable = TF_DataType.DtInvalid)
        {
            name = ops.name_scope("", "Variable", initial_value);
            _initial_value = ops.convert_to_tensor(initial_value, name: "initial_value");
        }
    }
}
