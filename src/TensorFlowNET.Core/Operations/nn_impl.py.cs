using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class nn_impl : Python
    {
        /// <summary>
        /// Calculate the mean and variance of `x`
        /// </summary>
        /// <param name="x"></param>
        /// <param name="axes"></param>
        /// <param name="name"></param>
        /// <param name="keep_dims"></param>
        /// <returns></returns>
        public (Tensor, Tensor) moments(Tensor x, 
            int[] axes,
            string name = null,
            bool keep_dims = false)
        {
            with<ops.name_scope>(new ops.name_scope(name, "moments", new { x, axes }), scope =>
            {
                var y = math_ops.cast(x, TF_DataType.TF_FLOAT);
                // mean = math_ops.reduce_mean(y, axes, keepdims = True, name = "mean")

            });

            throw new NotImplementedException("");
        }
    }
}
