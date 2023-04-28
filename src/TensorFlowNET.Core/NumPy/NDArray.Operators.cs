using static Tensorflow.Binding;

namespace Tensorflow.NumPy
{
    public partial class NDArray
    {
        [AutoNumPy]
        public static NDArray operator +(NDArray lhs, NDArray rhs) => new NDArray(BinaryOpWrapper("add", lhs, rhs));
        [AutoNumPy]
        public static NDArray operator -(NDArray lhs, NDArray rhs) => new NDArray(BinaryOpWrapper("sub", lhs, rhs));
        [AutoNumPy]
        public static NDArray operator *(NDArray lhs, NDArray rhs) => new NDArray(BinaryOpWrapper("mul", lhs, rhs));
        [AutoNumPy]
        public static NDArray operator /(NDArray lhs, NDArray rhs) => new NDArray(BinaryOpWrapper("div", lhs, rhs));
        [AutoNumPy]
        public static NDArray operator %(NDArray lhs, NDArray rhs) => new NDArray(BinaryOpWrapper("mod", lhs, rhs));
        [AutoNumPy]
        public static NDArray operator >(NDArray lhs, NDArray rhs) => new NDArray(gen_math_ops.greater(lhs, rhs));
        [AutoNumPy]
        public static NDArray operator <(NDArray lhs, NDArray rhs) => new NDArray(gen_math_ops.less(lhs, rhs));
        [AutoNumPy]
        public static NDArray operator -(NDArray lhs) => new NDArray(gen_math_ops.neg(lhs));
        [AutoNumPy]
        public static NDArray operator ==(NDArray lhs, NDArray rhs)
        {
            if (ReferenceEquals(lhs, rhs))
                return Scalar(true);
            if (lhs is null)
                return Scalar(false);
            if (rhs is null)
                return Scalar(false);
            // TODO(Rinne): use np.allclose instead.
            if (lhs.dtype.is_floating() || rhs.dtype.is_floating())
            {
                var diff = tf.abs(lhs - rhs);
                return new NDArray(gen_math_ops.less(diff, new NDArray(1e-5).astype(diff.dtype)));
            }
            else
            {
                return new NDArray(math_ops.equal(lhs, rhs));
            }
        }
        [AutoNumPy]
        public static NDArray operator !=(NDArray lhs, NDArray rhs)
        {
            if (ReferenceEquals(lhs, rhs))
                return Scalar(false);
            if (lhs is null || rhs is null)
                return Scalar(true);
            if (lhs.dtype.is_floating() || rhs.dtype.is_floating())
            {
                var diff = tf.abs(lhs - rhs);
                return new NDArray(gen_math_ops.greater_equal(diff, new NDArray(1e-5).astype(diff.dtype)));
            }
            else
            {
                return new NDArray(math_ops.not_equal(lhs, rhs));
            }
        }
    }
}
