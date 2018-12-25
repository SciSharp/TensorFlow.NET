using Google.Protobuf.Collections;
using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
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
