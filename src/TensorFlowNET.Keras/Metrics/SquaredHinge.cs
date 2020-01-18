using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Metrics
{
    public class SquaredHinge : MeanMetricWrapper
    {
        public SquaredHinge(string name = "squared_hinge", string dtype = null)
            : base(Losses.Loss.squared_hinge, name, dtype)
        {
        }
    }
}
