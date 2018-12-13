using Google.Protobuf.Collections;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;
using tensor_shape_pb2 = Tensorflow;

namespace TensorFlowNET.Core
{
    public class TensorShape
    {
        private int[] _dims;

        public TensorShape()
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
