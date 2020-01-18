using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Metrics
{
    public class TruePositives : _ConfusionMatrixConditionCount
    {
        public TruePositives(float thresholds = 0.5F, string name = null, string dtype = null)
            : base(Utils.MetricsUtils.ConfusionMatrix.TRUE_POSITIVES, thresholds, name, dtype)
        {
        }
    }
}
