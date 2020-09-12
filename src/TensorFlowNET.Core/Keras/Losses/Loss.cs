using System;
using System.Collections.Generic;
using System.Text;

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

        public Loss(string reduction = ReductionV2.AUTO, string name = null)
        {
            this.reduction = reduction;
            this.name = name;
            _allow_sum_over_batch_size = false;
        }

        void _set_name_scope()
        {
            _name_scope = name;
        }
    }
}
