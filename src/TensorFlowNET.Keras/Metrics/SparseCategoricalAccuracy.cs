using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Metrics
{
    public class SparseCategoricalAccuracy : MeanMetricWrapper
    {
        public SparseCategoricalAccuracy(string name = "sparse_categorical_accuracy", string dtype = null)
            : base(Metric.sparse_categorical_accuracy, name, dtype)
        {
        }

    }
}
