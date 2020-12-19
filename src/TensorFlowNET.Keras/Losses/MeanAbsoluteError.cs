using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace Tensorflow.Keras.Losses
{
    public class MeanAbsoluteError : LossFunctionWrapper, ILossFunc
    {
        public MeanAbsoluteError(
            string reduction = null,
            string name = null) :
            base(reduction: reduction, name: name == null ? "mean_absolute_error" : name){ }

        public override Tensor Apply(Tensor y_true = null, Tensor y_pred =null, bool from_logits = false, int axis = -1)
        {
            Tensor y_pred_dispatch = ops.convert_to_tensor(y_pred);
            Tensor y_true_cast = gen_math_ops.cast(y_true, y_pred_dispatch.dtype);
            return  gen_math_ops.mean(math_ops.abs(y_pred_dispatch - y_true_cast), axis: -1);
        }
    }
}
