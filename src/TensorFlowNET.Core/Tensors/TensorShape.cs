using Google.Protobuf.Collections;
using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.Linq;
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

        /// <summary>
        /// Returns True iff `self` is fully defined in every dimension.
        /// </summary>
        /// <returns></returns>
        public bool is_fully_defined()
        {
            return Dimensions != null; 
        }
    }
}
