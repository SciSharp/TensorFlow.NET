using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Metrics
{
    public class LogCoshError : MeanMetricWrapper
    {
        public LogCoshError(string name = "logcosh", string dtype = null)
            : base(Losses.Loss.logcosh, name, dtype)
        {
        }
    }
}
