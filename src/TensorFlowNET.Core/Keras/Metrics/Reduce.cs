using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
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
        public Reduce(string reduction, string name, TF_DataType dtype = TF_DataType.DtInvalid)
            : base(name: name, dtype: dtype)
        {
            total = add_weight("total", initializer: tf.zeros_initializer);

            if (reduction == Reduction.WEIGHTED_MEAN ||
                reduction == Reduction.SUM_OVER_BATCH_SIZE)
            {
                count = add_weight("count", initializer: tf.zeros_initializer);
            }
        }
    }
}
