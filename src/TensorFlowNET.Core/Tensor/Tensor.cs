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

        private readonly int _ndim;
        public int ndim => _ndim;

        public Tensor(IntPtr handle)
        {
            _handle = handle;
            _dtype = c_api.TF_TensorType(_handle);
            _ndim = c_api.TF_NumDims(_handle);
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

        public T Data<T>()
        {
            /*var buffer = new byte[6 * sizeof(float)];
            var h1 = c_api.TF_TensorData(handle);
            var bytes = Marshal.PtrToStructure<float>(h1);
            Marshal.Copy(h1, buffer, 0, 24);*/

            return default(T);
        }
    }
}
