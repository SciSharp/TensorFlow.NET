using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Metrics
{
    public class Hinge : MeanMetricWrapper
    {
        public Hinge(string name = "hinge", string dtype = null)
            : base(Losses.Loss.hinge, name, dtype)
        {
        }
    }
}
