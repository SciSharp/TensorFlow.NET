using System;
using System.Linq;
using System.Text;
using NumSharp;
using Tensorflow.Contrib.Learn.Estimators;

namespace Tensorflow.Framework
{
    public static class tensor_shape
    {
        public static void assert_is_compatible_with(this Tensor self, Tensor other)
        {
            if (!self.is_compatible_with(other))
            {
                var selfDim = self.shape
                    .Aggregate(new StringBuilder("{"), (sb, i) => sb.Append(i).Append(", "), sb => sb.ToString())
                    .Replace(", }", "}");

                var otherDim = other.shape
                    .Aggregate(new StringBuilder("{"), (sb, i) => sb.Append(i).Append(", "), sb => sb.ToString())
                    .Replace(", }", "}");

                throw new ArgumentException($"Dimensions {selfDim} and {otherDim} are not compatible");
            }
        }

        public static Dimension dimension_at_index(TensorShape shape, int index)
        {
            return shape.rank < 0 ?
                new Dimension(-1) :
                new Dimension(shape.dims[index]);
        }

        public static int dimension_value(Dimension dimension)
            => dimension.value;

        public static TensorShape as_shape(this Shape shape)
             => new TensorShape(shape.Dimensions);
    }
}
