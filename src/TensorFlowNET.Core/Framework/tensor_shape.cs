using Tensorflow.NumPy;
using System;
using System.Linq;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow.Framework
{
    public static class tensor_shape
    {
        public static void assert_is_compatible_with(this Tensor self, Tensor other)
        {
            /*if (!self.is_compatible_with(other))
            {
                var selfDim = self.shape
                    .Aggregate(new StringBuilder("{"), (sb, i) => sb.Append(i).Append(", "), sb => sb.ToString())
                    .Replace(", }", "}");

                var otherDim = other.shape
                    .Aggregate(new StringBuilder("{"), (sb, i) => sb.Append(i).Append(", "), sb => sb.ToString())
                    .Replace(", }", "}");

                throw new ArgumentException($"Dimensions {selfDim} and {otherDim} are not compatible");
            }*/
        }

        public static bool is_compatible_with(this Tensor self, Tensor other)
        {
            bool _shape_is_compatible_0dim(Shape _this, Shape _other)
            {
                var __other = _other;
                if (_this.dims == null || __other.dims == null)
                    return true;

                if (_this.ndim != __other.ndim)
                    return false;

                foreach (var (x_dim, y_dim) in _this.dims.Zip(__other.dims, (x_dim, y_dim) => (x_dim, y_dim)))
                {
                    if (x_dim != y_dim)
                        return false;
                }

                return true;
            }

            if (other is SparseTensor)
            {
                return self.dtype.is_compatible_with(other.dtype);
            }

            return self.dtype.is_compatible_with(other.dtype) &&
                   _shape_is_compatible_0dim(self.shape, other.shape) &&
                   !(self is SparseTensor);
        }

        public static Dimension dimension_at_index(Shape shape, int index)
        {
            return shape.ndim < 0 ?
                new Dimension(-1) :
                new Dimension(shape.dims[index]);
        }

        public static int dimension_value(Dimension dimension)
            => (int)dimension.value;

        public static Shape most_specific_compatible_shape(this Shape self, Shape other)
        {
            var dims = range(self.ndim).Select(x => -1L).ToArray();
            foreach(var (i, (d1, d2)) in enumerate(zip(self.dims, other.dims)))
            {
                if (d1 == d2)
                    dims[i] = d1;
            }

            return new Shape(dims);
        }
    }
}
