namespace Tensorflow.Keras.Metrics;

public class Precision : Metric
{
    Tensor _thresholds;
    int _top_k;
    int _class_id;
    IVariableV1 true_positives;
    IVariableV1 false_positives;
    bool _thresholds_distributed_evenly;

    public Precision(float thresholds = 0.5f, int top_k = 0, int class_id = 0, string name = "recall", TF_DataType dtype = TF_DataType.TF_FLOAT)
        : base(name: name, dtype: dtype)
    {
        _thresholds = constant_op.constant(new float[] { thresholds });
        _top_k = top_k;
        _class_id = class_id;
        true_positives = add_weight("true_positives", shape: 1, initializer: tf.initializers.zeros_initializer());
        false_positives = add_weight("false_positives", shape: 1, initializer: tf.initializers.zeros_initializer());
    }

    public override Tensor update_state(Tensor y_true, Tensor y_pred, Tensor sample_weight = null)
    {
        return metrics_utils.update_confusion_matrix_variables(
            new Dictionary<string, IVariableV1> 
            {
                { "tp", true_positives },
                { "fp", false_positives },
            },
            y_true,
            y_pred,
            thresholds: _thresholds,
            thresholds_distributed_evenly: _thresholds_distributed_evenly,
            top_k: _top_k,
            class_id: _class_id,
            sample_weight: sample_weight);
    }

    public override Tensor result()
    {
        var result = tf.divide(true_positives.AsTensor(), tf.add(true_positives, false_positives));
        return  _thresholds.size == 1 ? result[0] : result;
    }

    public override void reset_states()
    {
        var num_thresholds = (int)_thresholds.size;
        keras.backend.batch_set_value(
            new List<(IVariableV1, NDArray)>
            {
                (true_positives, np.zeros(num_thresholds)),
                (false_positives, np.zeros(num_thresholds))
            });
    }
}
