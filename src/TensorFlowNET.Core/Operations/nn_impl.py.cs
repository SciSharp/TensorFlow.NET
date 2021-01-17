/*****************************************************************************
   Copyright 2018 The TensorFlow.NET Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

using Tensorflow.Operations;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class nn_impl
    {
        /// <summary>
        /// Normalizes along dimension `axis` using an L2 norm.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="axis"></param>
        /// <param name="epsilon"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor l2_normalize(Tensor x,
            int axis = 0,
            Tensor epsilon =null,
            string name = null)
        {
            return tf_with(ops.name_scope(name, "l2_normalize", new { x }), scope =>
            {
                x = ops.convert_to_tensor(x, name: "x");
                var sq = math_ops.square(x);
                var square_sum = math_ops.reduce_sum(sq, axis, keepdims: true);
                var x_inv_norm = math_ops.rsqrt(math_ops.maximum(square_sum, epsilon == null ? tf.Variable(1e-12f) : epsilon));
                return math_ops.multiply(x, x_inv_norm, name: name);
            });
        }

        /// <summary>
        /// Calculate the mean and variance of `x`
        /// </summary>
        /// <param name="x"> A `Tensor`.</param>
        /// <param name="axes"> Array of ints.  Axes along which to compute mean and variance.</param>
        /// <param name="name"> Name used to scope the operations that compute the moments.</param>
        /// <param name="keep_dims"> Produce moments with the same dimensionality as the input.</param>
        /// <returns> Two `Tensor` objects: `mean` and `variance`.</returns>
        public static (Tensor, Tensor) moments(Tensor x,
            int[] axes,
            string name = null,
            bool keep_dims = false)
        {
            return tf_with(ops.name_scope(name, "moments", new { x, axes }), scope =>
            {
                // The dynamic range of fp16 is too limited to support the collection of
                // sufficient statistics. As a workaround we simply perform the operations
                // on 32-bit floats before converting the mean and variance back to fp16
                var y = math_ops.cast(x, TF_DataType.TF_FLOAT);
                // Compute true mean while keeping the dims for proper broadcasting.
                var mean = math_ops.reduce_mean(y, axes, true, name = "mean");
                // Sample variance, not unbiased variance
                // Note: stop_gradient does not change the gradient that gets
                // backpropagated to the mean from the variance calculation,
                // because that gradient is zero
                var variance = math_ops.reduce_mean(math_ops.square_difference(y, array_ops.stop_gradient(mean)), axes, true, name = "Variance");
                if (!keep_dims)
                {
                    mean = array_ops.squeeze(mean, axes);
                    variance = array_ops.squeeze(variance, axes);
                }
                // TODO: if x.dtype == dtypes.float16:
                if (x.dtype == TF_DataType.TF_HALF)
                    return (math_ops.cast(mean, x.dtype), math_ops.cast(variance, x.dtype));
                else
                    return (mean, variance);
            });
        }

        /// <summary>
        /// Batch normalization.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="scale"></param>
        /// <param name="offset"></param>
        /// <param name="mean"></param>
        /// <param name="variance"></param>
        /// <param name="epsilon"></param>
        /// <param name="data_format"></param>
        /// <param name="is_training"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor[] fused_batch_norm(Tensor x,
            IVariableV1 scale,
            IVariableV1 offset,
            IVariableV1 mean,
            IVariableV1 variance,
            float epsilon = 0.001f,
            string data_format = "NHWC",
            bool is_training = true,
            string name = null,
            float exponential_avg_factor = 1.0f)
        {
            /*if (mean == null)
                mean = constant_op.constant(new float[0]);
            if (variance == null)
                variance = constant_op.constant(new float[0]);*/
            var min_epsilon = 1.001e-5f;
            epsilon = epsilon > min_epsilon ? epsilon : min_epsilon;

            var results = gen_nn_ops.fused_batch_norm_v3(x,
                scale,
                offset,
                mean,
                variance,
                epsilon: epsilon,
                exponential_avg_factor: exponential_avg_factor,
                data_format: data_format,
                is_training: is_training,
                name: name);

            var y = results[0];
            var running_mean = results[1];
            var running_var = results[2];

            return new[] { y, running_mean, running_var };
        }

        /// <summary>
        /// Same as math_ops.count_nonzero.
        /// The reduction is done in dtype, which can be faster for 32-bit dtypes.
        /// </summary>
        /// <param name="input_tensor">The numeric tensor.</param>
        /// <param name="dtype">The reduction dtype.</param>
        /// <returns>number of nonzero values with type dtype</returns>
        private static Tensor _count_nonzero(Tensor input_tensor, TF_DataType dtype = TF_DataType.TF_INT64)
        {
            return tf_with(ops.name_scope("count_nonzero", "count_nonzero", new { input_tensor }), scope =>
            {
                var zero = array_ops.zeros(new NumSharp.Shape(), dtype: input_tensor.dtype);
                var nonzero_count = math_ops.reduce_sum(
                math_ops.cast(gen_math_ops.not_equal(input_tensor, zero), dtype: dtype), name: "nonzero_count");
                return nonzero_count;
            });
        }

        public static Tensor sigmoid_cross_entropy_with_logits(Tensor labels, Tensor logits, string name = null)
        {
            return tf_with(ops.name_scope(name, "logistic_loss", new { logits, labels }), scope =>
            {
                name = scope;
                logits = ops.convert_to_tensor(logits, name: "logits");
                labels = ops.convert_to_tensor(labels, name: "labels");
                labels.TensorShape.merge_with(logits.TensorShape);

                var zeros = array_ops.zeros_like(logits, dtype: logits.dtype);
                var cond = (logits >= zeros);
                var relu_logits = array_ops.where(cond, logits, zeros);
                var neg_abs_logits = array_ops.where(cond, -logits, logits);

                return math_ops.add(
                    relu_logits - logits * labels,
                    gen_math_ops.log1p(gen_math_ops.exp(neg_abs_logits)),
                    name: name);
            });
        }

        /// <summary>
        /// Returns the fraction of zeros in value.
        /// </summary>
        /// <param name="value">A tensor of numeric type.</param>
        /// <param name="name">A name for the operation (optional).</param>
        /// <returns>The fraction of zeros in value, with type float32.</returns>
        public static Tensor zero_fraction(Tensor value, string name = null)
        {
            return tf_with(ops.name_scope(name, "zero_fraction", new { value }), scope =>
            {
                value = ops.convert_to_tensor(value, name: "value");
                Tensor size = array_ops.size(value, out_type: dtypes.int64);
                Tensor zero_fraction_float32 = null;

                size = gen_math_ops.less_equal(size, dtypes.int32.max());
                Tensor num_nonzero = control_flow_ops.cond(
                        size,
                        () => math_ops.cast(_count_nonzero(value, dtype: dtypes.int32), TF_DataType.TF_INT64),
                        () => _count_nonzero(value, dtype: dtypes.int64)
                        );

                tf_with(ops.name_scope("counts_to_fraction"), count_scope =>
                {
                    var num_zero = math_ops.subtract(math_ops.cast(size, TF_DataType.TF_INT64), num_nonzero);
                    var num_zero_float32 = math_ops.cast(num_zero, dtype: dtypes.float32);
                    var size_float32 = math_ops.cast(size, dtype: dtypes.float32);
                    zero_fraction_float32 = num_zero_float32 / size_float32;
                });

                return array_ops.identity(zero_fraction_float32, "fraction");
            });
        }
    }
}
