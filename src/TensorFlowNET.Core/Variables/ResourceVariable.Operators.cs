using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.NumPy;

namespace Tensorflow
{
    public partial class ResourceVariable
    {
        public static Tensor operator +(ResourceVariable x, int y) => x.value() + y;
        public static Tensor operator +(ResourceVariable x, float y) => x.value() + y;
        public static Tensor operator +(ResourceVariable x, double y) => x.value() + y;
        public static Tensor operator +(ResourceVariable x, ResourceVariable y) => x.value() + y.value();
        public static Tensor operator -(ResourceVariable x, int y) => x.value() - y;
        public static Tensor operator -(ResourceVariable x, float y) => x.value() - y;
        public static Tensor operator -(ResourceVariable x, double y) => x.value() - y;
        public static Tensor operator -(ResourceVariable x, Tensor y) => x.value() - y;
        public static Tensor operator -(ResourceVariable x, ResourceVariable y) => x.value() - y.value();

        public static Tensor operator *(ResourceVariable x, ResourceVariable y) => x.value() * y.value();
        public static Tensor operator *(ResourceVariable x, Tensor y) => x.value() * y;
        public static Tensor operator *(ResourceVariable x, NDArray y) => x.value() * y;

        public static Tensor operator <(ResourceVariable x, Tensor y) => x.value() < y;

        public static Tensor operator >(ResourceVariable x, Tensor y) => x.value() > y;
    }
}
