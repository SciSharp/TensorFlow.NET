using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Losses;
using Tensorflow.Keras.Utils;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Metrics
{
    /// <summary>
    /// Encapsulates metrics that perform a reduce operation on the values.
    /// </summary>
    public class Reduce : Metric
    {
        IVariableV1 total;
        IVariableV1 count;
        string _reduction;
        TF_DataType _dtype;

        public Reduce(string reduction, string name, TF_DataType dtype = TF_DataType.DtInvalid)
            : base(name: name, dtype: dtype)
        {
            _reduction = reduction;
            _dtype = dtype;
            total = add_weight("total", initializer: tf.zeros_initializer);

            if (reduction == Reduction.WEIGHTED_MEAN ||
                reduction == Reduction.SUM_OVER_BATCH_SIZE)
            {
                count = add_weight("count", initializer: tf.zeros_initializer);
            }
        }

        public Tensor update_state(Tensor values, Tensor sample_weight = null)
        {
            if(sample_weight != null)
            {
                (values, sample_weight) = losses_utils.squeeze_or_expand_dimensions(
                    values, sample_weight: sample_weight);

                sample_weight = math_ops.cast(sample_weight, dtype: values.dtype);
                values = math_ops.multiply(values, sample_weight);
            }

            Tensor update_total_op = null;
            var value_sum = math_ops.reduce_sum(values);
            tf_with(ops.control_dependencies(new[] { value_sum }), ctl =>
            {
                var update_total_op = total.assign_add(value_sum);
            });

            Tensor num_values = null;
            if (_reduction == ReductionV2.WEIGHTED_MEAN)
            {
                if (sample_weight == null)
                    num_values = math_ops.cast(array_ops.size(values), _dtype);
                else
                    num_values = math_ops.reduce_sum(sample_weight);
            }

            return tf_with(ops.control_dependencies(new[] { update_total_op }), ctl 
                => count.assign_add(num_values));
        }
    }
}
