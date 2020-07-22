using NumSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow.Eager
{
    public partial class EagerTensor
    {
        public override string ToString()
        {
            switch (rank)
            {
                case -1:
                    return $"tf.Tensor: shape={TensorShape}, dtype={dtype.as_numpy_name()}, numpy={GetFormattedString(dtype, numpy())}";
                case 0:
                    return $"tf.Tensor: shape={TensorShape}, dtype={dtype.as_numpy_name()}, numpy={GetFormattedString(dtype, numpy())}";
                default:
                    return $"tf.Tensor: shape={TensorShape}, dtype={dtype.as_numpy_name()}, numpy={GetFormattedString(dtype, numpy())}";
            }
        }

        public static string GetFormattedString(TF_DataType dtype, NDArray nd)
        {
            if (nd.size == 0)
                return "[]";

            switch (dtype)
            {
                case TF_DataType.TF_STRING:
                    return string.Join(string.Empty, nd.ToArray<byte>()
                        .Select(x => x < 32 || x > 127 ? "\\x" + x.ToString("x") : Convert.ToChar(x).ToString()));
                case TF_DataType.TF_BOOL:
                    return (nd.GetByte(0) > 0).ToString();
                case TF_DataType.TF_VARIANT:
                case TF_DataType.TF_RESOURCE:
                    return "<unprintable>";
                default:
                    return nd.ToString();
            }
        }
    }
}
