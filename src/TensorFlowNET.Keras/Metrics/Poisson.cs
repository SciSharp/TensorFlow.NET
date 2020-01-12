using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Metrics
{
    public class Poisson : MeanMetricWrapper
    {
        public Poisson(string name = "logcosh", string dtype = null)
            : base(Losses.Loss.logcosh, name, dtype)
        {
        }
    }
}
