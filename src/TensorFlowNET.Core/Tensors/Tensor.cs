//using Newtonsoft.Json;
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
    public partial class Tensor : Python, IDisposable, ITensorOrOperation
    {
        private readonly IntPtr _handle;

        private int _id;
        //[JsonIgnore]
        public int Id => _id;
        //[JsonIgnore]
        public Graph graph => op?.graph;
        //[JsonIgnore]
        public Operation op { get; }
        //[JsonIgnore]
        public Tensor[] outputs => op.outputs;

        /// <summary>
        /// The string name of this tensor.
        /// </summary>
        public string name => $"{(op == null ? "Operation was not named" : $"{op.name}:{value_index}")}";

        public int value_index { get; }

        private Status status = new Status();

        private TF_DataType _dtype = TF_DataType.DtInvalid;
        public TF_DataType dtype => _handle == IntPtr.Zero ? _dtype : c_api.TF_TensorType(_handle);
        public ulong bytesize => _handle == IntPtr.Zero ? 0 : c_api.TF_TensorByteSize(_handle);
        public ulong itemsize => _handle == IntPtr.Zero ? 0 : c_api.TF_DataTypeSize(dtype);
        public ulong size => _handle == IntPtr.Zero ? 0 : bytesize / itemsize;
        public IntPtr buffer => _handle == IntPtr.Zero ? IntPtr.Zero : c_api.TF_TensorData(_handle);
        public int num_consumers(TF_Output oper_out) => _handle == IntPtr.Zero ? 0 : c_api.TF_OperationOutputNumConsumers(oper_out);

        private TF_Output? _tf_output;

        public long[] shape
        {
            get
            {
                var dims = new long[rank < 0 ? 0 : rank];

                if (_handle == IntPtr.Zero)
                {
                    c_api.TF_GraphGetTensorShape(op.graph, _as_tf_output(), dims, rank, status);
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
                if (value == null)
                    c_api.TF_GraphSetTensorShape(this.graph, this._as_tf_output(), null, -1, status);
                else
                    c_api.TF_GraphSetTensorShape(this.graph, this._as_tf_output(), value, value.Length, status);
            }
        }

        public int[] _shape_tuple()
        {
            if (shape == null) return null;
            return shape.Select(x => (int)x).ToArray();
        }

        public TensorShape getShape()
        {
            return tensor_util.to_shape(shape);
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
                    return c_api.TF_GraphGetTensorNumDims(op.graph, output, status);
                }
                else
                {
                    return c_api.TF_NumDims(_handle);
                }
            }
        }

        public int NDims => rank;

        //[JsonIgnore]
        public Operation[] Consumers => consumers();

        public string Device => op.Device;

        public Operation[] consumers()
        {
            var output = _as_tf_output();
            var consumer_names = c_api.TF_OperationOutputConsumers_wrapper(output);
            return consumer_names.Select(x => graph.OperationByName(x)).ToArray();
        }

        public TF_Output _as_tf_output()
        {
            if(!_tf_output.HasValue)
                _tf_output = new TF_Output(op, value_index);

            return _tf_output.Value;
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
                data[i] = Marshal.PtrToStructure<T>(buffer + (int)(i * itemsize));
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
        public NDArray eval(params FeedItem[] feed_dict)
        {
            return ops._eval_using_default_session(this, feed_dict, graph);
        }

        public NDArray eval(Session session, FeedItem[] feed_dict = null)
        {
            return ops._eval_using_default_session(this, feed_dict, graph, session);
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
                case "Byte":
                case "String":
                    return TF_DataType.TF_STRING;
                default:
                    throw new NotImplementedException("ToTFDataType error");
            }
        }

        public Tensor this[int slice_spec]
        {
            get
            {
                var slice_spec_s = new int[] { slice_spec };
                var begin = new List<int>();
                var end = new List<int>();
                var strides = new List<int>();

                var index = 0;
                var (new_axis_mask, shrink_axis_mask) = (0, 0);
                var (begin_mask, end_mask) = (0, 0);
                var ellipsis_mask = 0;

                foreach(var s in slice_spec_s)
                {
                    {
                        begin.Add(s);
                        end.Add(s + 1);
                        strides.Add(1);
                        shrink_axis_mask |= (1 << index);
                    }
                    
                    index += 1;
                }

                return with(ops.name_scope(null, "strided_slice", new { begin, end, strides }), scope =>
                {
                    string name = scope;
                    if(begin != null)
                    {
                        var (packed_begin, packed_end, packed_strides) =
                            (array_ops.stack(begin.ToArray()),
                            array_ops.stack(end.ToArray()),
                            array_ops.stack(strides.ToArray()));

                        return gen_array_ops.strided_slice(
                            this,
                            packed_begin,
                            packed_end,
                            packed_strides,
                            begin_mask: begin_mask,
                            end_mask: end_mask,
                            shrink_axis_mask: shrink_axis_mask,
                            new_axis_mask: new_axis_mask,
                            ellipsis_mask: ellipsis_mask,
                            name: name);
                    }

                    throw new NotImplementedException("");
                });
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

            return $"tf.Tensor '{name}' shape=({string.Join(",", shape)}) dtype={dtype.ToString()}";
        }

        public void Dispose()
        {
            c_api.TF_DeleteTensor(_handle);
            status.Dispose();
        }
    }
}
