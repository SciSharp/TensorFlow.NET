using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow.NumPy
{
    public partial class NDArray
    {
        public static NDArray operator +(NDArray lhs, NDArray rhs) => new NDArray(BinaryOpWrapper("add", lhs, rhs));
        public static NDArray operator -(NDArray lhs, NDArray rhs) => new NDArray(BinaryOpWrapper("sub", lhs, rhs));
        public static NDArray operator *(NDArray lhs, NDArray rhs) => new NDArray(BinaryOpWrapper("mul", lhs, rhs));
        public static NDArray operator /(NDArray lhs, NDArray rhs) => new NDArray(BinaryOpWrapper("div", lhs, rhs));
        public static NDArray operator >(NDArray lhs, NDArray rhs) => new NDArray(gen_math_ops.greater(lhs, rhs));
        public static NDArray operator <(NDArray lhs, NDArray rhs) => new NDArray(gen_math_ops.less(lhs, rhs));
        public static NDArray operator -(NDArray lhs) => new NDArray(gen_math_ops.neg(lhs));
    }
}
