using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Metrics
{
    public class MeanAbsolutePercentageError : MeanMetricWrapper
    {
        public MeanAbsolutePercentageError(string name = "mean_absolute_percentage_error", string dtype = null)
            : base(Losses.Loss.mean_absolute_percentage_error, name, dtype)
        {
        }
    }
}
