using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace Tensorflow.Keras.Losses
{
    public class CosineSimilarity : LossFunctionWrapper, ILossFunc
    {
        protected int axis=-1;
        public CosineSimilarity(
            string reduction = null,
            int axis=-1,
            string name = null) :
            base(reduction: reduction, name: name == null ? "cosine_similarity" : name)
        {
            this.axis = axis;
        }

        public override Tensor Apply(Tensor y_true = null, Tensor y_pred =null, bool from_logits = false, int axis = -1)
        {
            Tensor y_true_normalize = nn_impl.l2_normalize(y_true, axis : this.axis);
            Tensor y_pred_normalize = nn_impl.l2_normalize(y_pred, axis: this.axis);
            return -math_ops.reduce_sum(y_true_normalize * y_pred_normalize, axis : this.axis);
        }
    }
}
