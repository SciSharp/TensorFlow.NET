using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Metrics
{
    public class MeanAbsoluteError : MeanMetricWrapper
    {
        public MeanAbsoluteError(string name = "mean_absolute_error", string dtype = null)
            : base(Losses.Loss.mean_absolute_error, name, dtype)
        {
        }
    }
}
