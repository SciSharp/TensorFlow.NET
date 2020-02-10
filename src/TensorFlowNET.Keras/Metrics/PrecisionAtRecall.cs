using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Metrics
{
    public class PrecisionAtRecall : SensitivitySpecificityBase
    {
        public PrecisionAtRecall(float recall, int num_thresholds = 200, string name = null, string dtype = null) : base(recall, num_thresholds, name, dtype)
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
