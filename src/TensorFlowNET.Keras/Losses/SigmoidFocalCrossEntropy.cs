using static HDF.PInvoke.H5L.info_t;

namespace Tensorflow.Keras.Losses;

public class SigmoidFocalCrossEntropy : LossFunctionWrapper
{
    float _alpha;
    float _gamma;

    public SigmoidFocalCrossEntropy(bool from_logits = false,
        float alpha = 0.25f,
        float gamma = 2.0f,
        string reduction = "none",
        string name = "sigmoid_focal_crossentropy") :
        base(reduction: reduction,
             name: name, 
             from_logits: from_logits)
    {
        _alpha = alpha;
        _gamma = gamma;
    }

    public override Tensor Apply(Tensor y_true, Tensor y_pred, bool from_logits = false, int axis = -1)
    {
        y_true = tf.cast(y_true, dtype: y_pred.dtype);
        var ce = keras.backend.binary_crossentropy(y_true, y_pred, from_logits: from_logits);
        var pred_prob = from_logits ? tf.sigmoid(y_pred) : y_pred;

        var p_t = (y_true * pred_prob) + ((1f - y_true) * (1f - pred_prob));
        Tensor alpha_factor = constant_op.constant(1.0f);
        Tensor modulating_factor = constant_op.constant(1.0f);

        if(_alpha > 0)
        {
            var alpha = tf.cast(constant_op.constant(_alpha), dtype: y_true.dtype);
            alpha_factor = y_true * alpha + (1f - y_true) * (1f - alpha);
        }

        if (_gamma > 0)
        {
            var gamma = tf.cast(constant_op.constant(_gamma), dtype: y_true.dtype);
            modulating_factor = tf.pow(1f - p_t, gamma);
        }

        return tf.reduce_sum(alpha_factor * modulating_factor * ce, axis = -1);
    }
}
