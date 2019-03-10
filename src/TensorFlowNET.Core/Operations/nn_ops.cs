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

        /// <summary>
        /// Adds `bias` to `value`.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="bias"></param>
        /// <param name="data_format"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor bias_add(Tensor value, 
            RefVariable bias, 
            string data_format = null, 
            string name = null)
        {
            return Python.with(ops.name_scope(name, "BiasAdd", new { value, bias }), scope =>
            {
                value = ops.convert_to_tensor(value, name: "input");
                var bias_tensor = ops.convert_to_tensor(bias, dtype: value.dtype, name: "bias");
                return gen_nn_ops.bias_add(value, bias_tensor, data_format: data_format, name: name);
            });
        }
    }
}
