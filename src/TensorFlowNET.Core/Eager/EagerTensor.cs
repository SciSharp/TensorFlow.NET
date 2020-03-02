using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Eager
{
    public class EagerTensor : Tensor
    {
        public EagerTensor(IntPtr handle) : base(handle)
        {
        }

        public EagerTensor(string value, string device_name) : base(value)
        {
        }

        public override string ToString()
        {
            switch (rank)
            {
                case -1:
                    return $"tf.Tensor: shape=<unknown>, dtype={dtype.as_numpy_name()}, numpy={GetFormattedString()}";
                case 0:
                    return $"tf.Tensor: shape=(), dtype={dtype.as_numpy_name()}, numpy={GetFormattedString()}";
                default:
                    return $"tf.Tensor: shape=({string.Join(",", shape)}), dtype={dtype.as_numpy_name()}, numpy={GetFormattedString()}";
            }
        }

        private string GetFormattedString()
        {
            var nd = numpy();
            switch (dtype)
            {
                case TF_DataType.TF_STRING:
                    return $"b'{(string)nd}'";
                default:
                    return nd.ToString();
            }
        }
    }
}
