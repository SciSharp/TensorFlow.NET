using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Metrics
{
    public class SumOverBatchSizeMetricWrapper : SumOverBatchSize
    {
        public SumOverBatchSizeMetricWrapper(Func<Tensor, Tensor, Tensor> fn, string name, string dtype = null)
        {
            throw new NotImplementedException();
        }

        public override void update_state(Args args, KwArgs kwargs)
        {
            throw new NotImplementedException();
        }

        public override Hashtable get_config()
        {
            throw new NotImplementedException();
        }
    }
}
