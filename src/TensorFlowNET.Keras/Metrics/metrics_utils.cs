using Tensorflow.NumPy;

namespace Tensorflow.Keras.Metrics;

public class metrics_utils
{
    public static Tensor sparse_top_k_categorical_matches(Tensor y_true, Tensor y_pred, int k = 5)
    {
        var reshape_matches = false;
        var y_true_rank = y_true.shape.ndim;
        var y_pred_rank = y_pred.shape.ndim;
        var y_true_org_shape = tf.shape(y_true);

        if (y_pred_rank > 2)
        {
            y_pred = tf.reshape(y_pred, (-1, y_pred.shape[-1]));
        }
            
        if (y_true_rank > 1)
        {
            reshape_matches = true;
            y_true = tf.reshape(y_true, new Shape(-1));
        }

        var matches = tf.cast(
            tf.math.in_top_k(
                predictions: y_pred, targets: tf.cast(y_true, np.int32), k: k
            ),
            dtype: keras.backend.floatx()
        );

        if (reshape_matches)
        {
            return tf.reshape(matches, shape: y_true_org_shape);
        }
        
        return matches;
    }
}
