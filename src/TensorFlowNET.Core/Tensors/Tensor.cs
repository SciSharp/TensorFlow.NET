using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using static Tensorflow.c_api;

namespace Tensorflow
{
    /// <summary>
    /// A tensor is a generalization of vectors and matrices to potentially higher dimensions. 
    /// Internally, TensorFlow represents tensors as n-dimensional arrays of base datatypes.
    /// </summary>
    public partial class Tensor : IDisposable
    {
        private readonly IntPtr _handle;

        private int _id;
        public int Id => _id;

        public Graph Graph => op?.Graph;
        public Operation op { get; }

        /// <summary>
        /// The string name of this tensor.
        /// </summary>
        public string name => $"{(op == null ? "Operation was not named" : $"{op.Name}:{value_index}")}";

        public int value_index { get; }

        private Status status = new Status();

        private TF_DataType _dtype = TF_DataType.DtInvalid;
        public TF_DataType dtype => _handle == IntPtr.Zero ? _dtype : c_api.TF_TensorType(_handle);
        public ulong bytesize => _handle == IntPtr.Zero ? 0 : c_api.TF_TensorByteSize(_handle);
        public ulong dataTypeSize => _handle == IntPtr.Zero ? 0 : c_api.TF_DataTypeSize(dtype);
        public ulong size => _handle == IntPtr.Zero ? 0 : bytesize / dataTypeSize;
        public IntPtr buffer => _handle == IntPtr.Zero ? IntPtr.Zero : c_api.TF_TensorData(_handle);
        public int num_consumers(TF_Output oper_out) => _handle == IntPtr.Zero ? 0 : c_api.TF_OperationOutputNumConsumers(oper_out);
        public long[] shape
        {
            get
            {
                var dims = new long[rank];

                if (_handle == IntPtr.Zero)
                {
                    c_api.TF_GraphGetTensorShape(op.Graph, _as_tf_output(), dims, rank, status);
                    status.Check();
                }
                else
                {
                    for (int i = 0; i < rank; i++)
                        dims[i] = c_api.TF_Dim(_handle, i);
                }

                return dims;
            }

            set
            {
                // c_api.TF_GraphSetTensorShape_wrapper
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
        public int rank
        {
            get
            {
                if (_handle == IntPtr.Zero)
                {
                    var output = _as_tf_output();
                    return c_api.TF_GraphGetTensorNumDims(op.Graph, output, status);
                }
                else
                {
                    return c_api.TF_NumDims(_handle);
                }
            }
        }

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
        }

        private IntPtr Allocate(NDArray nd)
        {
            IntPtr dotHandle = IntPtr.Zero;
            ulong size = 0;

            if (nd.dtype.Name != "String")
            {
                dotHandle = Marshal.AllocHGlobal(nd.dtypesize * nd.size);
                size = (ulong)(nd.size * nd.dtypesize);
            }
            
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
                    dotHandle = Marshal.AllocHGlobal(bytes.Length + 1);
                    Marshal.Copy(bytes, 0, dotHandle, bytes.Length);
                    size = (ulong)bytes.Length;
                    break;
                default:
                    throw new NotImplementedException("Marshal.Copy failed.");
            }

            var dataType = ToTFDataType(nd.dtype);
            // shape
            var dims = nd.shape.Select(x => (long)x).ToArray();
            // Free the original buffer and set flag
            Deallocator deallocator = (IntPtr values, IntPtr len, ref bool closure) =>
            {
                Marshal.FreeHGlobal(dotHandle);
                closure = true;
            };

            var tfHandle = c_api.TF_NewTensor(dataType,
                dims, 
                nd.ndim,
                dotHandle,
                size,
                deallocator,
                ref deallocator_called);

            return tfHandle;
        }

        public Tensor(Operation op, int value_index, TF_DataType dtype)
        {
            this.op = op;
            this.value_index = value_index;
            this._dtype = dtype;
            _id = ops.uid();
        }

        public List<Operation> consumers()
        {
            var output = _as_tf_output();
            var consumer_names = c_api.TF_OperationOutputConsumers_wrapper(output);
            return consumer_names.Select(x => Graph.OperationByName(x)).ToList();
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

        /// <summary>
        /// Evaluates this tensor in a `Session`.
        /// </summary>
        /// <param name="feed_dict">A dictionary that maps `Tensor` objects to feed values.</param>
        /// <param name="session">The `Session` to be used to evaluate this tensor.</param>
        /// <returns></returns>
        public NDArray eval(FeedItem[] feed_dict = null, Session session = null)
        {
            return ops._eval_using_default_session(this, feed_dict, Graph, session);
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

        public override string ToString()
        {
            if(NDims == 0)
            {
                switch (dtype)
                {
                    case TF_DataType.TF_INT32:
                        return Data<int>()[0].ToString();
                }
            }

            return $"{name} shape=({string.Join(",", shape)}) dtype={dtype.ToString()}";
        }

        public void Dispose()
        {
            c_api.TF_DeleteTensor(_handle);
            status.Dispose();
        }
    }
}
