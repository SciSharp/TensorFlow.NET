using System;
using System.Linq;

namespace Tensorflow
{
    public partial class TensorShape
    {
        public override bool Equals(Object obj)
        {
            switch (obj)
            {
                case TensorShape shape1:
                    if (rank == -1 && shape1.rank == -1)
                        return false;
                    else if (rank != shape1.rank)
                        return false;
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
