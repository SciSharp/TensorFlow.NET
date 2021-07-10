using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow.NumPy
{
    public partial class NDArray
    {
        public override bool Equals(object obj)
        {
            return obj switch
            {
                int val => GetAtIndex<int>(0) == val,
                long val => GetAtIndex<long>(0) == val,
                float val => GetAtIndex<float>(0) == val,
                double val => GetAtIndex<double>(0) == val,
                NDArray val => Equals(this, val),
                _ => base.Equals(obj)
            };
        }

        bool Equals(NDArray x, NDArray y)
        {
            if (x.ndim != y.ndim)
                return false;

            return Enumerable.SequenceEqual(x.ToByteArray(), y.ToByteArray());
        }
    }
}
