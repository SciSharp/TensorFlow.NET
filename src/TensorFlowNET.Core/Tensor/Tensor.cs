using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public class Tensor
    {
        public Operation op { get; }
        public int value_index { get; }
        public TF_DataType dtype { get; }

        public Graph graph => op.graph;

        public string name;
        public IntPtr handle { get; }
        public int ndim { get; }
        public ulong bytesize { get; }
        public ulong dataTypeSize { get;}
        public ulong size => bytesize / dataTypeSize;
        public IntPtr buffer { get; }

        public Tensor(IntPtr handle)
        {
            this.handle = handle;
            dtype = c_api.TF_TensorType(handle);
            ndim = c_api.TF_NumDims(handle);
            bytesize = c_api.TF_TensorByteSize(handle);
            buffer = c_api.TF_TensorData(handle);
            dataTypeSize = c_api.TF_DataTypeSize(dtype);
        }

        public Tensor(Operation op, int value_index, TF_DataType dtype)
        {
            this.op = op;
            this.value_index = value_index;
            this.dtype = dtype;
        }

        public TF_Output _as_tf_output()
        {
            return c_api_util.tf_output(op._c_op, value_index);
        }

        public T[] Data<T>()
        {
            var data = new T[size];

            for (ulong i = 0; i < size; i++)
            {
                data[i] = Marshal.PtrToStructure<T>(buffer + (int)(i * dataTypeSize));
            }

            return data;
        }

        public byte[] Data()
        {
            var data = new byte[bytesize];
            Marshal.Copy(buffer, data, 0, (int)bytesize);

            return data;
        }
    }
}
