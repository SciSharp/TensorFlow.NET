using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace Tensorflow.Keras.Losses
{
    public class CategoricalCrossentropy : LossFunctionWrapper, ILossFunc
    {
        float label_smoothing;
        public CategoricalCrossentropy(
            bool from_logits = false,
            float label_smoothing = 0,
            string reduction = null,
            string name = null) :
            base(reduction: reduction,
                 name: name == null ? "categorical_crossentropy" : name, 
                 from_logits: from_logits)
        {
            this.label_smoothing = label_smoothing;
        }


        public override Tensor Apply(Tensor y_true, Tensor y_pred, bool from_logits = false, int axis = -1)
        {
            // Try to adjust the shape so that rank of labels = rank of logits - 1.
            return keras.backend.categorical_crossentropy(y_true, y_pred, from_logits: from_logits);
        }
    }
}
