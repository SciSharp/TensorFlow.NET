using System.Linq;
using NumSharp;
using Tensorflow.Framework;

namespace Tensorflow.Contrib.Learn.Estimators
{
    public static class tensor_signature
    {
        public static bool is_compatible_with(this Tensor self, Tensor other)
        {
            bool _shape_is_compatible_0dim(Shape _this, Shape _other)
            {
                var __other = tensor_shape.as_shape(_other);
                if (_this.Dimensions == null || __other.dims == null)
                    return true;

                if (_this.NDim != __other.ndim)
                    return false;

                foreach (var (x_dim, y_dim) in _this.Dimensions.Zip(__other.dims, (x_dim, y_dim) => (x_dim, y_dim)))
                {
                    if (x_dim != y_dim)
                        return false;
                }

                return true;
            }

            if (other.is_sparse())
            {
                return self.dtype.is_compatible_with(other.dtype);
            }

            return self.dtype.is_compatible_with(other.dtype) &&
                   _shape_is_compatible_0dim(self.shape, other.shape) &&
                   !self.is_sparse();
        }
    }
}
