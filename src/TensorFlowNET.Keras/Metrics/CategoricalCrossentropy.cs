using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Metrics
{
    public class CategoricalCrossentropy : MeanMetricWrapper
    {
        public CategoricalCrossentropy(string name = "categorical_crossentropy", string dtype = null, bool from_logits = false, float label_smoothing = 0)
            : base(Fn, name, dtype)
        {
        }

        internal static Tensor Fn(Tensor y_true, Tensor y_pred)
        {
            return Losses.Loss.categorical_crossentropy(y_true, y_pred);
        }
    }
}
