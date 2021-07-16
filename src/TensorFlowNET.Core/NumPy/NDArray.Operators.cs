using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow.NumPy
{
    public partial class NDArray
    {
        public static NDArray operator +(NDArray lhs, NDArray rhs) => lhs.Tensor + rhs.Tensor;
        public static NDArray operator -(NDArray lhs, NDArray rhs) => lhs.Tensor - rhs.Tensor;
        public static NDArray operator *(NDArray lhs, NDArray rhs) => lhs.Tensor * rhs.Tensor;
        public static NDArray operator /(NDArray lhs, NDArray rhs) => lhs.Tensor / rhs.Tensor;
        public static NDArray operator >(NDArray lhs, NDArray rhs) => lhs.Tensor > rhs.Tensor;
        public static NDArray operator <(NDArray lhs, NDArray rhs) => lhs.Tensor < rhs.Tensor;
    }
}
