using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Metrics
{
    public class FalsePositives : _ConfusionMatrixConditionCount
    {
        public FalsePositives(float thresholds = 0.5F, string name = null, string dtype = null) 
            : base(Utils.MetricsUtils.ConfusionMatrix.FALSE_POSITIVES, thresholds, name, dtype)
        {
        }
    }
}
