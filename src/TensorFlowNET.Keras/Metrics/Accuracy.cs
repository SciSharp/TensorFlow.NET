using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Metrics
{
    public class Accuracy : MeanMetricWrapper
    {
        public Accuracy(string name = "accuracy", string dtype = null)
            : base(Metric.accuracy, name, dtype)
        {
        }
    }
}
