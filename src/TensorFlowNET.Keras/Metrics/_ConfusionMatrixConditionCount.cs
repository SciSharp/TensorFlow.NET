using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Keras.Utils.MetricsUtils;

namespace Tensorflow.Keras.Metrics
{
    public class _ConfusionMatrixConditionCount : Metric
    {
        public _ConfusionMatrixConditionCount(string confusion_matrix_cond, float thresholds= 0.5f, string name= null, string dtype= null)
            : base(name, dtype)
        {
            throw new NotImplementedException();
        }

        public override Tensor result()
        {
            throw new NotImplementedException();
        }

        public override void update_state(Args args, KwArgs kwargs)
        {
            throw new NotImplementedException();
        }

        public override void reset_states()
        {
            throw new NotImplementedException();
        }

        public override Hashtable get_config()
        {
            throw new NotImplementedException();
        }
    }
}
