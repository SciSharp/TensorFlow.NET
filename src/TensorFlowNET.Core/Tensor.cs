using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;

namespace TensorFlowNET.Core
{
    public class Tensor
    {
        private Operation _op;
        private int _value_index;
        private DataType _dtype;

        public Tensor(Operation op, int value_index, DataType dtype)
        {
            _op = op;
            _value_index = value_index;
            _dtype = dtype;
        }
    }
}
