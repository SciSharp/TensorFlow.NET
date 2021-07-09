using System;
using System.Collections.Generic;
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
                _ => base.Equals(obj)
            };
        }
    }
}
