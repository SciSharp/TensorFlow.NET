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

    /// <summary>
    /// Computes how often targets are in the top K predictions.
    /// </summary>
    /// <param name="y_true"></param>
    /// <param name="y_pred"></param>
    /// <param name="k"></param>
    /// <returns></returns>
    IMetricFunc TopKCategoricalAccuracy(int k = 5, string name = "top_k_categorical_accuracy", TF_DataType dtype = TF_DataType.TF_FLOAT);

    /// <summary>
    /// Computes the recall of the predictions with respect to the labels.
    /// </summary>
    /// <param name="thresholds"></param>
    /// <param name="top_k"></param>
    /// <param name="class_id"></param>
    /// <param name="name"></param>
    /// <param name="dtype"></param>
    /// <returns></returns>
    IMetricFunc Recall(float thresholds = 0.5f, int top_k = 1, int class_id = 0, string name = "recall", TF_DataType dtype = TF_DataType.TF_FLOAT);
}
