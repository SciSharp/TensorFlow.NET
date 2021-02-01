using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Operations;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace Tensorflow.Keras.Losses
{
    public class LogCosh : LossFunctionWrapper, ILossFunc
    {
        public LogCosh(
            string reduction = null,
            string name = null) :
            base(reduction: reduction, name: name == null ? "log_cosh" : name){ }

        public override Tensor Apply(Tensor y_true = null, Tensor y_pred =null, bool from_logits = false, int axis = -1)
        {
            Tensor y_pred_dispatch = ops.convert_to_tensor(y_pred);
            Tensor y_true_cast = gen_math_ops.cast(y_true, y_pred_dispatch.dtype);
            Tensor x = y_pred_dispatch - y_true_cast;

            return gen_math_ops.mean(x + gen_math_ops.softplus(-2.0 * x) - math_ops.cast(math_ops.log(tf.Variable(2.0)), x.dtype), axis: -1);
        }
    }
}
