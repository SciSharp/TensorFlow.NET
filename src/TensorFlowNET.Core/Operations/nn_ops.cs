using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Operations;

namespace Tensorflow
{
    public class nn_ops
    {
        public static Convolution Convolution(TensorShape input_shape,
            TensorShape filter_shape,
            string padding,
            int[] strides,
            int[] dilation_rate,
            string name = null,
            string data_format = null) => new Convolution(input_shape,
                filter_shape,
                padding,
                strides,
                dilation_rate,
                name: name,
                data_format: data_format);
    }
}
