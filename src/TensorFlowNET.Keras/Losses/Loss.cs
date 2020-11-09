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
        string _name_scope;

        public string Reduction => reduction;

        public Loss(string reduction = ReductionV2.AUTO, string name = null)
        {
            this.reduction = reduction;
            this.name = name;
            _allow_sum_over_batch_size = false;
        }

        public virtual Tensor Apply(Tensor y_true, Tensor y_pred, bool from_logits = false, int axis = -1)
        {
            throw new NotImplementedException("");
        }

        public Tensor Call(Tensor y_true, Tensor y_pred)
        {
            var losses = Apply(y_true, y_pred);
            return losses_utils.compute_weighted_loss(losses, reduction: ReductionV2.SUM_OVER_BATCH_SIZE);
        }

        void _set_name_scope()
        {
            _name_scope = name;
        }
    }
}
