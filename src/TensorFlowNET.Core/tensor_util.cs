using System;
using System.Collections.Generic;
using System.Text;
using np = NumSharp.Core.NumPy;

namespace TensorFlowNET.Core
{
    public static class tensor_util
    {
        public static void make_tensor_proto(object values, Type dtype = null)
        {
            var nparray = np.array(values as Array, dtype);
        }
    }
}
