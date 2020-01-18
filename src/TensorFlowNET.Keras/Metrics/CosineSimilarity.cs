using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Metrics
{
    public class CosineSimilarity : MeanMetricWrapper
    {
        public CosineSimilarity(string name = "cosine_similarity", string dtype = null, int axis = -1)
            : base(Fn, name, dtype)
        {
        }

        internal static Tensor Fn(Tensor y_true, Tensor y_pred)
        {
            return Metric.cosine_proximity(y_true, y_pred);
        }
    }
}
