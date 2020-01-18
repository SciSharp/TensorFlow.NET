using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Metrics
{
    public class MeanSquaredError : MeanMetricWrapper
    {
        public MeanSquaredError(string name = "mean_squared_error", string dtype = null)
            : base(Losses.Loss.mean_squared_error, name, dtype)
        {
        }
    }
}
