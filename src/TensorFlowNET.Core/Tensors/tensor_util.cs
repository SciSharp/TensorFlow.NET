/*****************************************************************************
   Copyright 2018 The TensorFlow.NET Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

using NumSharp;
using System;
using System.Linq;
using NumSharp.Utilities;
using System.Text;
using Tensorflow.Eager;

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
        /// Returns the constant value of the given tensor, if efficiently calculable.
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="partial"></param>
        /// <returns></returns>
        public static NDArray constant_value(Tensor tensor, bool partial = false)
        {
            if (tensor is EagerTensor)
                return tensor.numpy();

            NDArray ret = _ConstantValue(tensor, partial);
            if (!(ret is null))
                tensor.graph.prevent_feeding(tensor);

            return ret;
        }

        private static NDArray _ConstantValue(Tensor tensor, bool partial)
        {
            if (tensor.op.type == "Const")
            {
                return MakeNdarray(tensor.op.get_attr("value") as TensorProto);
            }

            return null;
        }

        public static NDArray MakeNdarray(TensorProto tensor)
        {
            var shape = tensor.TensorShape.Dim.Select(x => (int)x.Size).ToArray();
            int num_elements = np.prod(shape);
            var tensor_dtype =  tensor.Dtype.as_numpy_dtype();

            if (tensor.TensorContent.Length > 0)
            {
                return np.frombuffer(tensor.TensorContent.ToByteArray(), tensor_dtype).reshape(shape);
            }
            else if (tensor.Dtype == DataType.DtHalf || tensor.Dtype == DataType.DtBfloat16)
#pragma warning disable CS0642 // Possible mistaken empty statement
                ;
#pragma warning restore CS0642 // Possible mistaken empty statement
            else if (tensor.Dtype == DataType.DtFloat)
#pragma warning disable CS0642 // Possible mistaken empty statement
                ;
#pragma warning restore CS0642 // Possible mistaken empty statement
            else if (new DataType[] { DataType.DtInt32, DataType.DtUint8 }.Contains(tensor.Dtype))
            {
                if (tensor.IntVal.Count == 1)
                    return np.repeat(np.array(tensor.IntVal[0]), num_elements).reshape(shape);
            }
            else if (tensor.Dtype == DataType.DtBool)
            {
                if (tensor.BoolVal.Count == 1)
                    return np.repeat(np.array(tensor.BoolVal[0]), num_elements).reshape(shape);
            }

            throw new NotImplementedException("MakeNdarray");
        }

        private static readonly TF_DataType[] quantized_types = new TF_DataType[]
        {
            TF_DataType.TF_QINT8, TF_DataType.TF_QUINT8, TF_DataType.TF_QINT16, TF_DataType.TF_QUINT16,
            TF_DataType.TF_QINT32
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

            // We first convert value to a numpy array or scalar.
            NDArray nparray = null;
            var np_dt = dtype.as_numpy_dtype();

            if (values is NDArray nd)
            {
                nparray = nd;
            }
            else
            {
                if (values == null)
                    throw new ValueError("None values not supported.");

                nparray = convert_to_numpy_ndarray(values);

                if (np_dt != null && np_dt != typeof(string))
                    nparray = nparray.astype(np_dt);
            }

            var numpy_dtype = nparray.dtype.as_dtype(dtype: dtype);
            if (numpy_dtype == TF_DataType.DtInvalid)
                throw new TypeError($"Unrecognized data type: {nparray.dtype}");

            // If dtype was specified and is a quantized type, we convert
            // numpy_dtype back into the quantized version.
            if (quantized_types.Contains(dtype))
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
                shape_size = new TensorShape(shape).size;
                is_same_size = shape_size == nparray.size;
            }

            var tensor_proto = new TensorProto
            {
                Dtype = numpy_dtype.as_datatype_enum(),
                TensorShape = tensor_util.as_shape(shape)
            };

            if (is_same_size && _TENSOR_CONTENT_TYPES.Contains(numpy_dtype) && shape_size > 1)
            {
                byte[] bytes = nparray.ToByteArray();
                tensor_proto.TensorContent = Google.Protobuf.ByteString.CopyFrom(bytes.ToArray());
                return tensor_proto;
            }

            if (numpy_dtype == TF_DataType.TF_STRING && !(values is NDArray))
            {
                if (values is string str)
                {
                    tensor_proto.StringVal.Add(Google.Protobuf.ByteString.CopyFromUtf8(str));
                    tensor_proto.TensorShape = tensor_util.as_shape(new int[0]);
                }
                else if (values is string[] str_values)
                    tensor_proto.StringVal.AddRange(str_values.Select(x => Google.Protobuf.ByteString.CopyFromUtf8(x)));
                else if(values is byte[] byte_values)
                    tensor_proto.TensorContent = Google.Protobuf.ByteString.CopyFrom(byte_values);
                
                return tensor_proto;
            }

            var proto_values = nparray.ravel();

            switch (nparray.dtype.Name)
            {
                case "Bool":
                case "Boolean":
                    tensor_proto.BoolVal.AddRange(proto_values.Data<bool>());
                    break;
                case "Int32":
                    tensor_proto.IntVal.AddRange(proto_values.Data<int>());
                    break;
                case "Int64":
                    tensor_proto.Int64Val.AddRange(proto_values.Data<long>());
                    break;
                case "Single":
                    tensor_proto.FloatVal.AddRange(proto_values.Data<float>());
                    break;
                case "Double":
                    tensor_proto.DoubleVal.AddRange(proto_values.Data<double>());
                    break;
                /*case "String":
                    tensor_proto.StringVal.AddRange(proto_values.Data<string>().Select(x => Google.Protobuf.ByteString.CopyFromUtf8(x.ToString())));
                    break;*/
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
                case bool boolVal:
                    nd = boolVal;
                    break;
                case int intVal:
                    nd = intVal;
                    break;
                case int[] intVals:
                    nd = np.array(intVals);
                    break;
                case int[,] intVals:
                    nd = np.array(intVals);
                    break;
                case long intVal:
                    nd = intVal;
                    break;
                case long[] intVals:
                    nd = np.array(intVals);
                    break;
                case long[,] intVals:
                    nd = np.array(intVals);
                    break;
                case float floatVal:
                    nd = floatVal;
                    break;
                case float[] floatVals:
                    nd = floatVals;
                    break;
                case float[,] floatVals:
                    nd = np.array(floatVals);
                    break;
                case double doubleVal:
                    nd = doubleVal;
                    break;
                case double[] doubleVals:
                    nd = np.array(doubleVals);
                    break;
                case double[,] doubleVals:
                    nd = np.array(doubleVals);
                    break;
                case string strVal:
                    nd = new NDArray(Encoding.ASCII.GetBytes(strVal));
                    break;
                case string[] strVals:
                    nd = strVals;
                    break;
                case byte[] byteValues:
                    nd = byteValues;
                    break;
                case byte[,] byteValues:
                    nd = np.array(byteValues);
                    break;
                default:
                    throw new NotImplementedException($"convert_to_numpy_ndarray: Support for type {values.GetType()} Not Implemented");
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
                // dim.Name = $"dim_{i}";

                shape.Dim.Add(dim);
            }

            return shape;
        }

        public static TensorShape to_shape(long[] dims)
        {
            return new TensorShape(dims.Select(x => (int)x).ToArray());
        }

        public static TensorShape to_shape(int[] dims)
        {
            return new TensorShape(dims);
        }

        public static TensorShape as_shape(this Shape shape)
        {
            return new TensorShape(shape.Dimensions);
        }

        public static TensorShape reshape(this Shape shape, int[] dims)
        {
            return new TensorShape(dims);
        }

        public static TensorShapeProto as_proto(this TensorShape tshape)
        {
            TensorShapeProto shape = new TensorShapeProto();

            for (int i = 0; i < tshape.ndim; i++)
            {
                var dim = new TensorShapeProto.Types.Dim();
                dim.Size = tshape.dims[i];
                //dim.Name = $"dim_{i}";

                shape.Dim.Add(dim);
            }

            return shape;
        }

        public static Tensor shape_tensor(int[] shape)
        {
            return ops.convert_to_tensor(shape, dtype: TF_DataType.TF_INT32, name: "shape");
        }
    }
}
