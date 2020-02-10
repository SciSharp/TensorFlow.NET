using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Metrics
{
    public class TopKCategoricalAccuracy : MeanMetricWrapper
    {
        public TopKCategoricalAccuracy(int k = 5, string name = "top_k_categorical_accuracy", string dtype = null)
            : base(Fn, name, dtype)
        {
        }

        internal static Tensor Fn(Tensor y_true, Tensor y_pred)
        {
            return Metric.top_k_categorical_accuracy(y_true, y_pred);
        }
    }
}
