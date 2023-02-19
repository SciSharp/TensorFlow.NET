namespace Tensorflow.Keras.Metrics;

public interface IMetricsApi
{
    Tensor binary_accuracy(Tensor y_true, Tensor y_pred);

    Tensor categorical_accuracy(Tensor y_true, Tensor y_pred);

    Tensor mean_absolute_error(Tensor y_true, Tensor y_pred);

    Tensor mean_absolute_percentage_error(Tensor y_true, Tensor y_pred);

    /// <summary>
    /// Calculates how often predictions matches integer labels.
    /// </summary>
    /// <param name="y_true">Integer ground truth values.</param>
    /// <param name="y_pred">The prediction values.</param>
    /// <returns>Sparse categorical accuracy values.</returns>
    Tensor sparse_categorical_accuracy(Tensor y_true, Tensor y_pred);

    /// <summary>
    /// Computes how often targets are in the top `K` predictions.
    /// </summary>
    /// <param name="y_true"></param>
    /// <param name="y_pred"></param>
    /// <param name="k"></param>
    /// <returns></returns>
    Tensor top_k_categorical_accuracy(Tensor y_true, Tensor y_pred, int k = 5);
}
