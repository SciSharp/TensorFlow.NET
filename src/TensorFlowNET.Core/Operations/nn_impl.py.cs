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
        /// <param name="x"> A `Tensor`.</param>
        /// <param name="axes"> Array of ints.  Axes along which to compute mean and variance.</param>
        /// <param name="name"> Name used to scope the operations that compute the moments.</param>
        /// <param name="keep_dims"> Produce moments with the same dimensionality as the input.</param>
        /// <returns> Two `Tensor` objects: `mean` and `variance`.</returns>
        public Tuple<Tensor, Tensor> moments(Tensor x, 
            int[] axes,
            string name = null,
            bool keep_dims = false)
        {
            Tuple<Tensor, Tensor> t = null;
            with(new ops.name_scope(name, "moments", new { x, axes }), scope =>
            {
                // The dynamic range of fp16 is too limited to support the collection of
                // sufficient statistics. As a workaround we simply perform the operations
                // on 32-bit floats before converting the mean and variance back to fp16
                var y = math_ops.cast(x, TF_DataType.TF_FLOAT);
                // Compute true mean while keeping the dims for proper broadcasting.
                var mean = math_ops.reduce_mean(y, axes, keep_dims = true, name = "mean");
                // Sample variance, not unbiased variance
                // Note: stop_gradient does not change the gradient that gets
                // backpropagated to the mean from the variance calculation,
                // because that gradient is zero
                var variance = math_ops.reduce_mean(math_ops.reduce_mean(math_ops.square_difference(y, array_ops.stop_gradient(mean)), axes, keep_dims = true, name = "Variance"));
                if (!keep_dims)
                {
                    mean = array_ops.squeeze(mean, axes);
                    variance = array_ops.squeeze(variance, axes);
                }
                // TODO: if x.dtype == dtypes.float16:
                if (x.dtype == TF_DataType.TF_FLOAT)
                {
                    t = Tuple.Create(math_ops.cast(mean, x.dtype), math_ops.cast(variance, x.dtype));
                    return;
                }
                else {
                    t = Tuple.Create(mean, variance);
                }
            });
            return t;
        }
    }
}
