using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace Tensorflow.Keras.Losses
{
    public class MeanAbsolutePercentageError : LossFunctionWrapper, ILossFunc
    {
        public MeanAbsolutePercentageError(
            string reduction = null,
            string name = null) :
            base(reduction: reduction, name: name == null ? "mean_absolute_percentage_error" : name){ }

        public override Tensor Apply(Tensor y_true = null, Tensor y_pred =null, bool from_logits = false, int axis = -1)
        {
            Tensor y_pred_dispatch = ops.convert_to_tensor(y_pred);
            Tensor y_true_cast = gen_math_ops.cast(y_true, y_pred_dispatch.dtype);
            Tensor diff = math_ops.abs(y_true_cast - y_pred_dispatch) / gen_math_ops.maximum(math_ops.abs(y_true_cast), gen_math_ops.cast(tf.constant(1e-7), y_pred_dispatch.dtype));
            return gen_math_ops.cast(tf.constant(100), y_pred_dispatch.dtype) * gen_math_ops.mean(diff, axis: -1);
        }
    }
}
