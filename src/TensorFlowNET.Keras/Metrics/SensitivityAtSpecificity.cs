using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Metrics
{
    public class SensitivityAtSpecificity : SensitivitySpecificityBase
    {
        public SensitivityAtSpecificity(float specificity, int num_thresholds = 200, string name = null, string dtype = null) : base(specificity, num_thresholds, name, dtype)
        {
            throw new NotImplementedException();
        }

        public override Tensor result()
        {
            throw new NotImplementedException();
        }

        public override Hashtable get_config()
        {
            throw new NotImplementedException();
        }
    }
}
