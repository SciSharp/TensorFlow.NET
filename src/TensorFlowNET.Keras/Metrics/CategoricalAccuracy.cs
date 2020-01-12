using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Metrics
{
    public class CategoricalAccuracy : MeanMetricWrapper
    {
        public CategoricalAccuracy(string name = "categorical_accuracy", string dtype = null)
            : base(Metric.categorical_accuracy, name, dtype)
        {
        }
    }
}
