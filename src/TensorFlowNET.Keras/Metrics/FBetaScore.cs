namespace Tensorflow.Keras.Metrics;

public class FBetaScore : Metric
{
    int _num_classes;
    string? _average;
    Tensor _beta;
    Tensor _threshold;
    Axis _axis;
    int[] _init_shape;

    IVariableV1 true_positives;
    IVariableV1 false_positives;
    IVariableV1 false_negatives;
    IVariableV1 weights_intermediate;

    public FBetaScore(int num_classes,
        string? average = null,
        float beta = 0.1f,
        float? threshold = null,
        string name = "fbeta_score",
        TF_DataType dtype = TF_DataType.TF_FLOAT)
        : base(name: name, dtype: dtype)
    {
        _num_classes = num_classes;
        _average = average;
        _beta = constant_op.constant(beta);
        _dtype = dtype;

        if (threshold.HasValue)
        {
            _threshold = constant_op.constant(threshold);
        }
        
        _init_shape = new int[0];

        if (average != "micro")
        {
            _axis = 0;
            _init_shape = new int[] { num_classes };
        }

        true_positives = add_weight("true_positives", shape: _init_shape, initializer: tf.initializers.zeros_initializer());
        false_positives = add_weight("false_positives", shape: _init_shape, initializer: tf.initializers.zeros_initializer());
        false_negatives = add_weight("false_negatives", shape: _init_shape, initializer: tf.initializers.zeros_initializer());
        weights_intermediate = add_weight("weights_intermediate", shape: _init_shape, initializer: tf.initializers.zeros_initializer());
    }

    public override Tensor update_state(Tensor y_true, Tensor y_pred, Tensor sample_weight = null)
    {
        if (_threshold == null)
        {
            _threshold = tf.reduce_max(y_pred, axis: -1, keepdims: true);
            // make sure [0, 0, 0] doesn't become [1, 1, 1]
            // Use abs(x) > eps, instead of x != 0 to check for zero
            y_pred = tf.logical_and(y_pred >= _threshold, tf.abs(y_pred) > 1e-12f);
        }
        else
        {
            y_pred = y_pred > _threshold;
        }

        y_true = tf.cast(y_true, _dtype);
        y_pred = tf.cast(y_pred, _dtype);

        true_positives.assign_add(_weighted_sum(y_pred * y_true, sample_weight));
        false_positives.assign_add(
            _weighted_sum(y_pred * (1 - y_true), sample_weight)
        );
        false_negatives.assign_add(
            _weighted_sum((1 - y_pred) * y_true, sample_weight)
        );
        weights_intermediate.assign_add(_weighted_sum(y_true, sample_weight));

        return weights_intermediate.AsTensor();
    }

    Tensor _weighted_sum(Tensor val, Tensor? sample_weight = null)
    {
        if (sample_weight != null)
        {
            val = tf.math.multiply(val, tf.expand_dims(sample_weight, 1));
        }
                
        return tf.reduce_sum(val, axis: _axis);
    }

    public override Tensor result()
    {
        var precision = tf.math.divide_no_nan(
            true_positives.AsTensor(), true_positives.AsTensor() + false_positives.AsTensor()
        );
        var recall = tf.math.divide_no_nan(
            true_positives.AsTensor(), true_positives.AsTensor() + false_negatives.AsTensor()
        );

        var mul_value = precision * recall;
        var add_value = (tf.math.square(_beta) * precision) + recall;
        var mean = tf.math.divide_no_nan(mul_value, add_value);
        var f1_score = mean * (1 + tf.math.square(_beta));

        Tensor weights;
        if (_average == "weighted")
        {
            weights = tf.math.divide_no_nan(
                weights_intermediate.AsTensor(), tf.reduce_sum(weights_intermediate.AsTensor())
            );
            f1_score = tf.reduce_sum(f1_score * weights);
        }
        // micro, macro
        else if (_average != null)
        {
            f1_score = tf.reduce_mean(f1_score);
        }

        return f1_score;
    }

    public override void reset_states()
    {
        var reset_value = np.zeros(_init_shape, dtype: _dtype);
        keras.backend.batch_set_value(
            new List<(IVariableV1, NDArray)>
            {
                (true_positives, reset_value),
                (false_positives, reset_value),
                (false_negatives, reset_value),
                (weights_intermediate, reset_value)
            });
    }
}
