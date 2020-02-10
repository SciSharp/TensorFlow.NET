using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Metrics
{
    public class TrueNegatives : _ConfusionMatrixConditionCount
    {
        public TrueNegatives(float thresholds = 0.5F, string name = null, string dtype = null)
            : base(Utils.MetricsUtils.ConfusionMatrix.TRUE_NEGATIVES, thresholds, name, dtype)
        {
        }
    }
}
