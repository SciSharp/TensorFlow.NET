using NumSharp.Core;
using NumSharp.Core.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using tensor_pb2 = Tensorflow;

namespace Tensorflow
{
    public static class tensor_util
    {
        public static TF_DataType[] _TENSOR_CONTENT_TYPES =
        {
            TF_DataType.TF_FLOAT, TF_DataType.TF_DOUBLE, TF_DataType.TF_INT32, TF_DataType.TF_UINT8, TF_DataType.TF_INT16,
            TF_DataType.TF_INT8, TF_DataType.TF_INT64, TF_DataType.TF_QINT8, TF_DataType.TF_QUINT8, TF_DataType.TF_QINT16,
            TF_DataType.TF_QUINT16, TF_DataType.TF_QINT32, TF_DataType.TF_UINT32, TF_DataType.TF_UINT64
        };

        /// <summary>
        /// Create a TensorProto.
        /// </summary>
        /// <param name="values"></param>
        /// <param name="dtype"></param>
        /// <param name="shape"></param>
        /// <param name="verify_shape"></param>
        /// <param name="allow_broadcast"></param>
        /// <returns></returns>
        public static TensorProto make_tensor_proto(object values, TF_DataType dtype = TF_DataType.DtInvalid, int[] shape = null, bool verify_shape = false, bool allow_broadcast = false)
        {
            if (allow_broadcast && verify_shape)
                throw new ValueError("allow_broadcast and verify_shape are not both allowed.");
            if (values is TensorProto tp)
                return tp;

            if (dtype != TF_DataType.DtInvalid)
                ;

            bool is_quantized = new TF_DataType[]
            {
                TF_DataType.TF_QINT8, TF_DataType.TF_QUINT8, TF_DataType.TF_QINT16, TF_DataType.TF_QUINT16,
                TF_DataType.TF_QINT32
            }.Contains(dtype);

            // We first convert value to a numpy array or scalar.
            NDArray nparray = null;

            if (values is NDArray nd)
            {
                nparray = nd;
            }
            else
            {
                if (values == null)
                    throw new ValueError("None values not supported.");

                switch (values)
                {
                    /*case bool boolVal:
                        nparray = boolVal;
                        break;*/
                    case int intVal:
                        nparray = intVal;
                        break;
                    case int[] intVals:
                        nparray = np.array(intVals);
                        break;
                    case float floatVal:
                        nparray = floatVal;
                        break;
                    case double doubleVal:
                        nparray = doubleVal;
                        break;
                    case string strVal:
                        nparray = strVal;
                        break;
                    default:
                        throw new Exception("make_tensor_proto Not Implemented");
                }
            }

            var numpy_dtype = dtypes.as_dtype(nparray.dtype);
            if (numpy_dtype == TF_DataType.DtInvalid)
                throw new TypeError($"Unrecognized data type: {nparray.dtype}");

            // If dtype was specified and is a quantized type, we convert
            // numpy_dtype back into the quantized version.
            if (is_quantized)
                numpy_dtype = dtype;

            bool is_same_size = false;
            int shape_size = 0;

            // If shape is not given, get the shape from the numpy array.
            if (shape == null)
            {
                shape = nparray.shape;
                is_same_size = true;
                shape_size = nparray.size;
            }
            else
            {
                throw new NotImplementedException("make_tensor_proto shape not implemented");
            }

            var tensor_proto = new tensor_pb2.TensorProto
            {
                Dtype = numpy_dtype.as_datatype_enum(),
                TensorShape = tensor_util.as_shape(shape)
            };

            if (is_same_size && _TENSOR_CONTENT_TYPES.Contains(numpy_dtype) && shape_size > 1)
            {
                var bytes = new List<byte>();
                var nd2 = nparray.ravel();
                switch (nparray.dtype.Name)
                {
                    case "Int32":
                        nd2.Data<int>().Select(x => 
                        {
                            bytes.AddRange(BitConverter.GetBytes(x));
                            return x;
                        }).ToArray();
                        break;
                    case "Single":
                        nd2.Data<float>().Select(x =>
                        {
                            bytes.AddRange(BitConverter.GetBytes(x));
                            return x;
                        }).ToArray();
                        break;
                    case "Double":
                        nd2.Data<double>().Select(x =>
                        {
                            bytes.AddRange(BitConverter.GetBytes(x));
                            return x;
                        }).ToArray();
                        break;
                    default:
                        throw new Exception("make_tensor_proto Not Implemented");
                }
                tensor_proto.TensorContent = Google.Protobuf.ByteString.CopyFrom(bytes.ToArray());
                return tensor_proto;
            }

            if (numpy_dtype == TF_DataType.TF_STRING && !(values is NDArray) && values is string str)
            {
                tensor_proto.StringVal.Add(Google.Protobuf.ByteString.CopyFromUtf8(str));
                return tensor_proto;
            }

            var proto_values = nparray.ravel();

            switch (nparray.dtype.Name)
            {
                case "Bool":
                    tensor_proto.BoolVal.AddRange(proto_values.Data<bool>());
                    break;
                case "Int32":
                    tensor_proto.IntVal.AddRange(proto_values.Data<int>());
                    break;
                case "Single":
                    tensor_proto.FloatVal.AddRange(proto_values.Data<float>());
                    break;
                case "Double":
                    tensor_proto.DoubleVal.AddRange(proto_values.Data<double>());
                    break;
                case "String":
                    tensor_proto.StringVal.AddRange(proto_values.Data<string>().Select(x => Google.Protobuf.ByteString.CopyFromUtf8(x.ToString())));
                    break;
                default:
                    throw new Exception("make_tensor_proto Not Implemented");
            }

            return tensor_proto;
        }

        public static NDArray convert_to_numpy_ndarray(object values)
        {
            NDArray nd;

            switch (values)
            {
                case NDArray val:
                    nd = val;
                    break;
                case int val:
                    nd = np.asarray(val);
                    break;
                case int[] val:
                    nd = np.array(val);
                    break;
                case float val:
                    nd = np.asarray(val);
                    break;
                case double val:
                    nd = np.asarray(val);
                    break;
                case string val:
                    nd = np.asarray(val);
                    break;
                default:
                    throw new Exception("Not Implemented");
            }

            return nd;
        }

        public static TensorShapeProto as_shape<T>(T[] dims)
        {
            TensorShapeProto shape = new TensorShapeProto();

            for (int i = 0; i < dims.Length; i++)
            {
                var dim = new TensorShapeProto.Types.Dim();
                switch(dims[i])
                {
                    case int n:
                        dim.Size = n;
                        break;
                    case long l:
                        dim.Size = l;
                        break;
                    default:
                        throw new NotImplementedException("as_shape Not Implemented");
                }
                dim.Name = $"dim_{i}";

                shape.Dim.Add(dim);
            }

            return shape;
        }

        public static TensorShape to_shape(long[] dims)
        {
            return new TensorShape(dims.Select(x => (int)x).ToArray());
        }

        public static TensorShape as_shape(this IShape shape)
        {
            return new TensorShape(shape.Dimensions);
        }

        public static TensorShape reshape(this IShape shape, int[] dims)
        {
            return new TensorShape(dims);
        }

        public static TensorShapeProto as_proto(this TensorShape tshape)
        {
            TensorShapeProto shape = new TensorShapeProto();

            for (int i = 0; i < tshape.NDim; i++)
            {
                var dim = new TensorShapeProto.Types.Dim();
                dim.Size = tshape.Dimensions[i];
                dim.Name = $"dim_{i}";

                shape.Dim.Add(dim);
            }

            return shape;
        }
    }
}
