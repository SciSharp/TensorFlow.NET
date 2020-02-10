using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Metrics
{
    public class FalseNegatives : _ConfusionMatrixConditionCount
    {
        public FalseNegatives(float thresholds = 0.5F, string name = null, string dtype = null)
            : base(Utils.MetricsUtils.ConfusionMatrix.FALSE_NEGATIVES, thresholds, name, dtype)
        {
        }
    }
}
