using Google.Protobuf.Collections;
using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;
using tensor_shape_pb2 = Tensorflow;

namespace TensorFlowNET.Core
{
    public class TensorShape : Shape
    {
        public TensorShape(params int[] shape) : base(shape)
        {

        }

        public TensorShape as_shape()
        {
            return this;
        }

        public TensorShapeProto as_proto()
        {
            TensorShapeProto dim = new TensorShapeProto();

            return new TensorShapeProto(dim);
        }
    }
}
