using NumSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Tensorflow
{
    public partial class TensorShape
    {
        public override bool Equals(Object obj)
        {
            switch (obj)
            {
                case TensorShape shape1:
                    return Enumerable.SequenceEqual(shape1.dims, dims);
                default:
                    return false;
            }
        }

        /*public static bool operator ==(TensorShape shape1, TensorShape shape2)
        {
            return false;
        }

        public static bool operator !=(TensorShape shape1, TensorShape shape2)
        {
            return false;
        }*/
    }
}
