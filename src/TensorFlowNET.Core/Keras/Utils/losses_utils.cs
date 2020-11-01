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

using System;
using System.Linq;
using Tensorflow.Keras.Losses;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Utils
{
    public class losses_utils
    {
        public static Tensor compute_weighted_loss(Tensor losses, Tensor sample_weight = null, string reduction = null, string name = null)
        {
            if (sample_weight == null)
                sample_weight = tf.constant(1.0f);
            var weighted_losses = scale_losses_by_sample_weight(losses, sample_weight);
            // Apply reduction function to the individual weighted losses.
            var loss = reduce_weighted_loss(weighted_losses, reduction);
            // Convert the result back to the input type.
            // loss = math_ops.cast(loss, losses.dtype);
            return loss;
        }

        public static Tensor scale_losses_by_sample_weight(Tensor losses, Tensor sample_weight)
        {
            // losses = math_ops.cast(losses, dtypes.float32);
            // sample_weight = math_ops.cast(sample_weight, dtypes.float32);
            // Update dimensions of `sample_weight` to match with `losses` if possible.
            // (losses, sample_weight) = squeeze_or_expand_dimensions(losses, sample_weight);
            return math_ops.multiply(losses, sample_weight);
        }

        public static (Tensor, Tensor) squeeze_or_expand_dimensions(Tensor y_pred, Tensor sample_weight)
        {
            var weights_shape = sample_weight.TensorShape;
            var weights_rank = weights_shape.ndim;
            if (weights_rank == 0)
                return (y_pred, sample_weight);
            throw new NotImplementedException("");
        }

        public static Tensor reduce_weighted_loss(Tensor weighted_losses, string reduction)
        {
            if (reduction == ReductionV2.NONE)
                return weighted_losses;
            else
            {
                var loss = math_ops.reduce_sum(weighted_losses);
                if (reduction == ReductionV2.SUM_OVER_BATCH_SIZE)
                    loss = _safe_mean(loss, _num_elements(weighted_losses));
                return loss;
            }
        }

        static Tensor _safe_mean(Tensor losses, Tensor num_present)
        {
            var total_loss = math_ops.reduce_sum(losses);
            return math_ops.div_no_nan(total_loss, num_present, name: "value");
        }

        static Tensor _num_elements(Tensor losses)
        {
            return tf_with(ops.name_scope("num_elements"), scope =>
            {
                return math_ops.cast(array_ops.size(losses, name: scope), dtype: losses.dtype);
            });
        }
    }
}
