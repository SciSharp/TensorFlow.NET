using System;
using Tensorflow.Keras.Utils;

namespace Tensorflow.Keras.Losses
{
    /// <summary>
    /// Loss base class.
    /// </summary>
    public abstract class Loss
    {
        protected string reduction;
        protected string name;
        bool _allow_sum_over_batch_size;
        protected bool from_logits = false;
        string _name_scope;

        public string Reduction => reduction;
        public string Name => name;
        public Loss(string reduction = ReductionV2.AUTO, 
            string name = null,
            bool from_logits = false)
        {
            this.reduction = reduction == null ? ReductionV2.SUM_OVER_BATCH_SIZE : reduction;
            this.name = name;
            this.from_logits = from_logits;
            _allow_sum_over_batch_size = false;
        }

        public virtual Tensor Apply(Tensor y_true, Tensor y_pred, bool from_logits = false, int axis = -1)
        {
            throw new NotImplementedException("");
        }

        public Tensor Call(Tensor y_true, Tensor y_pred, Tensor sample_weight = null)
        {
            var losses = Apply(y_true, y_pred, from_logits: from_logits);
            return losses_utils.compute_weighted_loss(losses, reduction: this.reduction , sample_weight: sample_weight);
        }

        void _set_name_scope()
        {
            _name_scope = name;
        }
    }
}
