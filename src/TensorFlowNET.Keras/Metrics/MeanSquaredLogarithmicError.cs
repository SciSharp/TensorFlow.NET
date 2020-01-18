using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Metrics
{
    public class MeanSquaredLogarithmicError : MeanMetricWrapper
    {
        public MeanSquaredLogarithmicError(string name = "mean_squared_logarithmic_error", string dtype = null)
            : base(Losses.Loss.mean_squared_logarithmic_error, name, dtype)
        {
        }
    }
}
