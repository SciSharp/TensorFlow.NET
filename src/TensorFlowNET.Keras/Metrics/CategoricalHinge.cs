using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Metrics
{
    public class CategoricalHinge : MeanMetricWrapper
    {
        public CategoricalHinge(string name = "categorical_hinge", string dtype = null)
            : base(Losses.Loss.categorical_hinge, name, dtype)
        {
        }
    }
}
