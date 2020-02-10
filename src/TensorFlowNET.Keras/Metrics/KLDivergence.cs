using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Metrics
{
    public class KLDivergence : MeanMetricWrapper
    {
        public KLDivergence(string name = "kullback_leibler_divergence", string dtype = null)
            : base(Losses.Loss.logcosh, name, dtype)
        {
        }
    }
}
