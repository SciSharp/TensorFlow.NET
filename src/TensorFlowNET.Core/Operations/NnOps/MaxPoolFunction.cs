using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Operations
{
    public class MaxPoolFunction : Python, IPoolFunction
    {
        public Tensor Apply(Tensor value,
            int[] ksize,
            int[] strides,
            string padding,
            string data_format = "NHWC",
            string name = null)
        {
            return with(ops.name_scope(name, "MaxPool", new { value }), scope => {

                value = ops.convert_to_tensor(value, name: "input");
                return gen_nn_ops.max_pool(
                    value,
                    ksize: ksize,
                    strides: strides,
                    padding: padding,
                    data_format: data_format,
                    name: name);
            });
        }
    }
}
