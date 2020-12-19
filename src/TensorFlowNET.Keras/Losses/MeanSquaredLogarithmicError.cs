using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace Tensorflow.Keras.Losses
{
    public class MeanSquaredLogarithmicError : LossFunctionWrapper, ILossFunc
    {
        public MeanSquaredLogarithmicError(
            string reduction = null,
            string name = null) :
            base(reduction: reduction, name: name == null ? "mean_squared_logarithmic_error" : name){ }


        public override Tensor Apply(Tensor y_true = null, Tensor y_pred =null, bool from_logits = false, int axis = -1)
        {
            Tensor y_pred_dispatch = ops.convert_to_tensor(y_pred);
            Tensor y_true_cast = gen_math_ops.cast(y_true, y_pred_dispatch.dtype);
            Tensor first_log=null, second_log=null;
            if (y_pred_dispatch.dtype == TF_DataType.TF_DOUBLE) {
                first_log = math_ops.log(gen_math_ops.maximum(y_pred_dispatch, 1e-7) + 1.0);
                second_log = math_ops.log(gen_math_ops.maximum(y_true_cast, 1e-7) + 1.0);
            }
            else {
                first_log = math_ops.log(gen_math_ops.maximum(y_pred_dispatch, 1e-7f) + 1.0f);
                second_log = math_ops.log(gen_math_ops.maximum(y_true_cast, 1e-7f) + 1.0f);
            }
            return gen_math_ops.mean(gen_math_ops.squared_difference(first_log, second_log), axis: -1);
        }
    }
}
