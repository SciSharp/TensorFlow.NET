using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.Text;
using tensor_pb2 = Tensorflow;

namespace Tensorflow
{
    public static class tensor_util
    {
        public static TensorProto make_tensor_proto(object values, TF_DataType dtype = TF_DataType.DtInvalid, Shape shape = null, bool verify_shape = false)
        {
            NDArray nparray;
            TensorProto tensor_proto = null;
            TF_DataType numpy_dtype;
            if(shape is null)
            {
                shape = new Shape();
            }

            switch (values)
            {
                case int val:
                    nparray = np.asarray(val);
                    numpy_dtype = dtypes.as_dtype(nparray.dtype);
                    tensor_proto = new tensor_pb2.TensorProto
                    {
                        Dtype = numpy_dtype.as_datatype_enum(),
                        TensorShape = shape.as_shape(nparray.shape).as_proto()
                    };
                    tensor_proto.IntVal.Add(val);
                    break;
                case float val:
                    nparray = np.asarray(val);
                    numpy_dtype = dtypes.as_dtype(nparray.dtype);
                    tensor_proto = new tensor_pb2.TensorProto
                    {
                        Dtype = numpy_dtype.as_datatype_enum(),
                        TensorShape = shape.as_shape(nparray.shape).as_proto()
                    };
                    tensor_proto.FloatVal.Add(val);
                    break;
                case double val:
                    nparray = np.asarray(val);
                    numpy_dtype = dtypes.as_dtype(nparray.dtype);
                    tensor_proto = new tensor_pb2.TensorProto
                    {
                        Dtype = numpy_dtype.as_datatype_enum(),
                        TensorShape = shape.as_shape(nparray.shape).as_proto()
                    };
                    tensor_proto.DoubleVal.Add(val);
                    break;
                case string val:
                    nparray = np.asarray(val);
                    numpy_dtype = dtypes.as_dtype(nparray.dtype);
                    tensor_proto = new tensor_pb2.TensorProto
                    {
                        Dtype = numpy_dtype.as_datatype_enum(),
                        TensorShape = shape.as_shape(nparray.shape).as_proto()
                    };
                    tensor_proto.StringVal.Add(Google.Protobuf.ByteString.CopyFrom(val, Encoding.UTF8));
                    break;
                default:
                    throw new Exception("Not Implemented");
            }

            return tensor_proto;
        }

        public static TensorShape as_shape(this Shape shape, int[] dims)
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
                dim.Name = $"{dim}_1";

                shape.Dim.Add(dim);
            }

            return shape;
        }
    }
}
