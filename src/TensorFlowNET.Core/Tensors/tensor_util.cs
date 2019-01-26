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
        public static TensorProto make_tensor_proto(NDArray nd, bool verify_shape = false)
        {
            var shape = nd.Storage.Shape;

            var numpy_dtype = dtypes.as_dtype(nd.dtype);
            var tensor_proto = new tensor_pb2.TensorProto
            {
                Dtype = numpy_dtype.as_datatype_enum(),
                TensorShape = shape.as_shape(nd.shape).as_proto()
            };

            switch (nd.dtype.Name)
            {
                case "Int32":
                    tensor_proto.IntVal.AddRange(nd.Data<int>());
                    break;
                case "Single":
                    tensor_proto.FloatVal.AddRange(nd.Data<float>());
                    break;
                case "Double":
                    tensor_proto.DoubleVal.AddRange(nd.Data<double>());
                    break;
                case "String":
                    tensor_proto.StringVal.AddRange(nd.Data<string>().Select(x => Google.Protobuf.ByteString.CopyFromUtf8(x)));
                    break;
                default:
                    throw new Exception("Not Implemented");
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

        public static TensorShapeProto as_shape(long[] dims)
        {
            TensorShapeProto shape = new TensorShapeProto();

            for (int i = 0; i < dims.Length; i++)
            {
                var dim = new TensorShapeProto.Types.Dim();
                dim.Size = dims[i];
                dim.Name = $"dim_{i}";

                shape.Dim.Add(dim);
            }

            return shape;
        }

        public static TensorShape to_shape(long[] dims)
        {
            return new TensorShape(dims.Select(x => (int)x).ToArray());
        }

        public static TensorShape as_shape(this IShape shape, int[] dims)
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
