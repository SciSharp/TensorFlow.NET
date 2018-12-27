using Google.Protobuf.Collections;
using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    /// <summary>
    /// Represents the shape of a `Tensor`.
    /// </summary>
    public class TensorShape : Shape
    {
        public TensorShape(params int[] dims) : base(dims)
        {

        }
    }
}
