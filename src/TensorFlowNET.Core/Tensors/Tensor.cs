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
    public partial class Tensor : IDisposable
    {
        private readonly IntPtr _handle;

        public Graph Graph => op.Graph;
        public Operation op { get; }

        public string name;
        public object value;
        public int value_index { get; }

        private TF_DataType _dtype = TF_DataType.DtInvalid;
        public TF_DataType dtype => _handle == IntPtr.Zero ? _dtype : c_api.TF_TensorType(_handle);
        public ulong bytesize => _handle == IntPtr.Zero ? 0 : c_api.TF_TensorByteSize(_handle);
        public ulong dataTypeSize => _handle == IntPtr.Zero ? 0 : c_api.TF_DataTypeSize(dtype);
        public ulong size => _handle == IntPtr.Zero ? 0 : bytesize / dataTypeSize;
        public IntPtr buffer => _handle == IntPtr.Zero ? IntPtr.Zero : c_api.TF_TensorData(_handle);
        public long[] shape
        {
            get
            {
                var dims = new long[rank];
                for (int i = 0; i < rank; i++)
                    dims[i] = c_api.TF_Dim(_handle, i);

                return dims;
            }
        }
        
        /// <summary>
        /// number of dimensions
        /// 0	Scalar (magnitude only)
        /// 1	Vector (magnitude and direction)
        /// 2	Matrix (table of numbers)
        /// 3	3-Tensor (cube of numbers)
        /// n	n-Tensor (you get the idea)
        /// </summary>
        public int rank => _handle == IntPtr.Zero ? 0 : c_api.TF_NumDims(_handle);
        public int NDims => rank;

        /// <summary>
        /// if original buffer is free.
        /// </summary>
        private bool deallocator_called;

        public Tensor(IntPtr handle)
        {
            _handle = handle;
        }

        public Tensor(NDArray nd)
        {
            _handle = Allocate(nd);
            value = nd.Data();
        }

        private IntPtr Allocate(NDArray nd)
        {
            var dotHandle = Marshal.AllocHGlobal(nd.dtypesize * nd.size);
            ulong size = (ulong)(nd.size * nd.dtypesize);

            switch (nd.dtype.Name)
            {
                case "Int16":
                    Marshal.Copy(nd.Data<short>(), 0, dotHandle, nd.size);
                    break;
                case "Int32":
                    Marshal.Copy(nd.Data<int>(), 0, dotHandle, nd.size);
                    break;
                case "Single":
                    Marshal.Copy(nd.Data<float>(), 0, dotHandle, nd.size);
                    break;
                case "Double":
                    Marshal.Copy(nd.Data<double>(), 0, dotHandle, nd.size);
                    break;
                case "String":
                    var value = nd.Data<string>()[0];
                    var bytes = Encoding.UTF8.GetBytes(value);
                    var buf = Marshal.AllocHGlobal(bytes.Length + 1);
                    Marshal.Copy(bytes, 0, buf, bytes.Length);

                    //c_api.TF_SetAttrString(op, "value", buf, (uint)bytes.Length);

                    size = (ulong)bytes.Length;
                    break;
                default:
                    throw new NotImplementedException("Marshal.Copy failed.");
            }

            var dataType = ToTFDataType(nd.dtype);
            
            var tfHandle = c_api.TF_NewTensor(dataType,
                nd.shape.Select(x => (long)x).ToArray(), // shape
                nd.ndim,
                dotHandle,
                size,
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
            this._dtype = dtype;
        }

        public TF_Output _as_tf_output()
        {
            return new TF_Output(op, value_index);
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

        public Tensor MaybeMove()
        {
            var tensor = c_api.TF_TensorMaybeMove(_handle);
            return tensor;
        }

        public TF_DataType ToTFDataType(Type type)
        {
            switch (type.Name)
            {
                case "Int16":
                    return TF_DataType.TF_INT16;
                case "Int32":
                    return TF_DataType.TF_INT32;
                case "Single":
                    return TF_DataType.TF_FLOAT;
                case "Double":
                    return TF_DataType.TF_DOUBLE;
                case "String":
                    return TF_DataType.TF_STRING;
                default:
                    throw new NotImplementedException("ToTFDataType error");
            }
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
