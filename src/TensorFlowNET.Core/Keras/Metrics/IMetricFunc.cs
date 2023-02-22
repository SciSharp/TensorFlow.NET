namespace Tensorflow.Keras.Metrics;

public interface IMetricFunc
{
    string Name { get; }
    /// <summary>
    /// Accumulates metric statistics.
    /// </summary>
    /// <param name="y_true"></param>
    /// <param name="y_pred"></param>
    /// <param name="sample_weight"></param>
    /// <returns></returns>
    Tensor update_state(Tensor y_true, Tensor y_pred, Tensor sample_weight = null);

    Tensor result();

    void reset_states();
}
