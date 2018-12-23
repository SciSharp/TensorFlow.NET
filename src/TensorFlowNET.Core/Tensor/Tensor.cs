using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public class Tensor
    {
        private readonly Operation _op;
        public Operation op => _op;
        private readonly int _value_index;
        public int value_index => _value_index;
        private TF_DataType _dtype;
        public TF_DataType dtype => _dtype;

        public Graph graph => _op.graph;

        public string name;

        private readonly IntPtr _handle;
        public IntPtr handle => _handle;

        private TF_Tensor tensor;

        public IntPtr buffer => c_api.TF_TensorData(tensor.buffer);

        public Tensor(IntPtr handle)
        {
            _handle = handle;
            tensor = Marshal.PtrToStructure<TF_Tensor>(handle);
            _dtype = tensor.dtype;
        }

        public Tensor(Operation op, int value_index, TF_DataType dtype)
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
