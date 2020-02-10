using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Metrics
{
    public class BinaryAccuracy : MeanMetricWrapper
    {
        public BinaryAccuracy(string name = "binary_accuracy", string dtype = null, float threshold = 0.5f)
            : base(Fn, name, dtype)
        {
        }

        internal static Tensor Fn(Tensor y_true, Tensor y_pred)
        {
            return Metric.binary_accuracy(y_true, y_pred);
        }
    }
}
