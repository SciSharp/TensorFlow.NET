using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class Tensor
    {
        private readonly Operation _op;
        public Operation op => _op;
        private readonly int _value_index;
        public int value_index => _value_index;
        private DataType _dtype;
        public DataType dtype => _dtype;

        public string name;

        public Tensor(Operation op, int value_index, DataType dtype)
        {
            _op = op;
            _value_index = value_index;
            _dtype = dtype;
        }

        public TF_Output _as_tf_output()
        {
            return c_api_util.tf_output(_op._c_op, _value_index);
        }
    }
}
