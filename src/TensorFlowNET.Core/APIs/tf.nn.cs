using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public static partial class tf
    {
        public static class nn
        {
            public static (Tensor, Tensor) moments(Tensor x,
            int[] axes,
            string name = null,
            bool keep_dims = false) => nn_impl.moments(x, axes, name: name, keep_dims: keep_dims);
        }
    }
}
