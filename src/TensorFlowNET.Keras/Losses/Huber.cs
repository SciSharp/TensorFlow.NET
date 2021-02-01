using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace Tensorflow.Keras.Losses
{
    public class Huber : LossFunctionWrapper, ILossFunc
    {
        protected Tensor delta = tf.Variable(1.0) ;
        public Huber (
            string reduction = null,
            Tensor delta = null,
            string name = null) :
            base(reduction: reduction, name: name == null ? "huber" : name)
        {
            this.delta = delta==null? this.delta: delta;
            
        }

        public override Tensor Apply(Tensor y_true = null, Tensor y_pred =null, bool from_logits = false, int axis = -1)
        {
            Tensor y_pred_cast = math_ops.cast(y_pred, dtype: TF_DataType.TF_FLOAT);
            Tensor y_true_cast = math_ops.cast(y_true, dtype: TF_DataType.TF_FLOAT);
            Tensor delta = math_ops.cast(this.delta, dtype: TF_DataType.TF_FLOAT);
            Tensor error = math_ops.subtract(y_pred_cast, y_true_cast);
            Tensor abs_error = math_ops.abs(error);
            Tensor half = ops.convert_to_tensor(0.5, dtype: abs_error.dtype);
            return gen_math_ops.mean(array_ops.where_v2(abs_error <= delta,
                                                        half * math_ops.pow(error, 2),
                                                        half * math_ops.pow(delta, 2) + delta * (abs_error - delta)),
                                     axis: -1);
        }
    }
}
