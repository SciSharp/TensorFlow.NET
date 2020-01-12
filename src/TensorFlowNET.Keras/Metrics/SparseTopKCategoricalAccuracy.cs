using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Metrics
{
    public class SparseTopKCategoricalAccuracy : MeanMetricWrapper
    {
        public SparseTopKCategoricalAccuracy(int k = 5, string name = "sparse_top_k_categorical_accuracy", string dtype = null)
            : base(Fn, name, dtype)
        {
            
        }

        internal static Tensor Fn(Tensor y_true, Tensor y_pred)
        {
            return Metric.sparse_top_k_categorical_accuracy(y_true, y_pred);
        }
    }
}
