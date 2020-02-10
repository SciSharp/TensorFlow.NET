using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Metrics
{
    public class SparseCategoricalCrossentropy : MeanMetricWrapper
    {
        public SparseCategoricalCrossentropy(string name = "sparse_categorical_crossentropy", string dtype = null, bool from_logits = false, int axis = -1)
            : base(Fn, name, dtype)
        {
        }

        internal static Tensor Fn(Tensor y_true, Tensor y_pred)
        {
            return Losses.Loss.sparse_categorical_crossentropy(y_true, y_pred);
        }
    }
}
