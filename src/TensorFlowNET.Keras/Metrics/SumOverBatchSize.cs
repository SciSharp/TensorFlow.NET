using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Metrics
{
    public class SumOverBatchSize : Reduce
    {
        public SumOverBatchSize(string name = "sum_over_batch_size", string dtype = null) : base(Reduction.SUM_OVER_BATCH_SIZE, name, dtype)
        {
        }
    }
}
