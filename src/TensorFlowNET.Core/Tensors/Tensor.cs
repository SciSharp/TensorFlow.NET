using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    /// <summary>
    /// A tensor is a generalization of vectors and matrices to potentially higher dimensions. 
    /// Internally, TensorFlow represents tensors as n-dimensional arrays of base datatypes.
    /// </summary>
    public class Tensor : IDisposable
    {
        private readonly IntPtr _handle;

        public Graph Graph => op.Graph;
        public Operation op { get; }

        public string name;
        public object value;
        public int value_index { get; }

        public TF_DataType dtype { get; }
        public ulong bytesize { get; }
        public ulong dataTypeSize { get;}
        public ulong size => bytesize / dataTypeSize;
        public IntPtr buffer { get; }
        public long[] shape { get; }
        
        /// <summary>
        /// number of dimensions
        /// 0	Scalar (magnitude only)
        /// 1	Vector (magnitude and direction)
        /// 2	Matrix (table of numbers)
        /// 3	3-Tensor (cube of numbers)
        /// n	n-Tensor (you get the idea)
        /// </summary>
        public int rank;
        public int NDims => rank;

        /// <summary>
        /// if original buffer is free.
        /// </summary>
        private bool deallocator_called;

        public Tensor(IntPtr handle)
        {
            _handle = handle;
            dtype = c_api.TF_TensorType(handle);
            rank = c_api.TF_NumDims(handle);
            bytesize = c_api.TF_TensorByteSize(handle);
            buffer = c_api.TF_TensorData(handle);
            dataTypeSize = c_api.TF_DataTypeSize(dtype);

            shape = new long[rank];
            for (int i = 0; i < rank; i++)
                shape[i] = c_api.TF_Dim(handle, i);
        }

        public Tensor(NDArray nd)
        {
            _handle = Allocate(nd);
            dtype = c_api.TF_TensorType(_handle);
            rank = c_api.TF_NumDims(_handle);
            bytesize = c_api.TF_TensorByteSize(_handle);
            buffer = c_api.TF_TensorData(_handle);
            dataTypeSize = c_api.TF_DataTypeSize(dtype);

            shape = new long[rank];
            for (int i = 0; i < rank; i++)
                shape[i] = c_api.TF_Dim(_handle, i);
        }

        private IntPtr Allocate(NDArray nd)
        {
            var dotHandle = Marshal.AllocHGlobal(nd.dtypesize * nd.size);

            switch (nd.dtype.Name)
            {
                case "Int32":
                    Marshal.Copy(nd.Data<int>(), 0, dotHandle, nd.size);
                    break;
                case "Single":
                    Marshal.Copy(nd.Data<float>(), 0, dotHandle, nd.size);
                    break;
                case "Double":
                    Marshal.Copy(nd.Data<double>(), 0, dotHandle, nd.size);
                    break;
                default:
                    throw new NotImplementedException("Marshal.Copy failed.");
            }

            var dataType = ToTFDataType(nd.dtype);
            var tfHandle = c_api.TF_NewTensor(dataType,
                nd.shape.Select(x => (long)x).ToArray(), // shape
                nd.ndim,
                dotHandle,
                (ulong)(nd.size * nd.dtypesize),
                (IntPtr values, IntPtr len, ref bool closure) =>
                {
                    // Free the original buffer and set flag
                    Marshal.FreeHGlobal(dotHandle);
                    closure = true;
                },
                ref deallocator_called);

            return tfHandle;
        }

        public Tensor(Operation op, int value_index, TF_DataType dtype)
        {
            this.op = op;
            this.value_index = value_index;
            this.dtype = dtype;
        }

        public TF_Output _as_tf_output()
        {
            return c_api_util.tf_output(op, value_index);
        }

        public T[] Data<T>()
        {
            // Column major order
            // https://en.wikipedia.org/wiki/File:Row_and_column_major_order.svg
            // matrix:[[1, 2, 3], [4, 5, 6]]
            // index:   0  2  4    1  3  5
            // result:  1  4  2    5  3  6
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

        public TF_DataType ToTFDataType(Type type)
        {
            switch (type.Name)
            {
                case "Int32":
                    return TF_DataType.TF_INT32;
                case "Single":
                    return TF_DataType.TF_FLOAT;
                case "Double":
                    return TF_DataType.TF_DOUBLE;
            }

            return TF_DataType.DtInvalid;
        }

        public void Dispose()
        {
            c_api.TF_DeleteTensor(_handle);
        }

        public static implicit operator IntPtr(Tensor tensor)
        {
            return tensor._handle;
        }

        public static implicit operator Tensor(IntPtr handle)
        {
            return new Tensor(handle);
        }
    }
}
