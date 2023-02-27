namespace Tensorflow.Keras.Metrics;

public interface IMetricsApi
{
    Tensor binary_accuracy(Tensor y_true, Tensor y_pred);

    Tensor categorical_accuracy(Tensor y_true, Tensor y_pred);
    Tensor categorical_crossentropy(Tensor y_true, Tensor y_pred, 
        bool from_logits = false, 
        float label_smoothing = 0f,
        Axis? axis = null);

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
    /// Computes the sparse categorical crossentropy loss.
    /// </summary>
    /// <param name="y_true"></param>
    /// <param name="y_pred"></param>
    /// <param name="from_logits"></param>
    /// <param name="ignore_class"></param>
    /// <param name="axis"></param>
    /// <returns></returns>
    Tensor sparse_categorical_crossentropy(Tensor y_true, Tensor y_pred, 
        bool from_logits = false,
        int? ignore_class = null,
        Axis? axis = null);

    /// <summary>
    /// Computes how often targets are in the top `K` predictions.
    /// </summary>
    /// <param name="y_true"></param>
    /// <param name="y_pred"></param>
    /// <param name="k"></param>
    /// <returns></returns>
    Tensor top_k_categorical_accuracy(Tensor y_true, Tensor y_pred, int k = 5);

    /// <summary>
    /// Calculates how often predictions equal labels.
    /// </summary>
    /// <returns></returns>
    IMetricFunc Accuracy(string name = "accuracy",
        TF_DataType dtype = TF_DataType.TF_FLOAT);

    /// <summary>
    /// Calculates how often predictions match binary labels.
    /// </summary>
    /// <returns></returns>
    IMetricFunc BinaryAccuracy(string name = "binary_accuracy",
        TF_DataType dtype = TF_DataType.TF_FLOAT,
        float threshold = 05f);

    /// <summary>
    /// Calculates how often predictions match one-hot labels.
    /// </summary>
    /// <returns></returns>
    IMetricFunc CategoricalCrossentropy(string name = "categorical_crossentropy", 
        TF_DataType dtype = TF_DataType.TF_FLOAT,
        bool from_logits = false,
        float label_smoothing = 0f,
        Axis? axis = null);

    /// <summary>
    /// Computes the crossentropy metric between the labels and predictions.
    /// </summary>
    /// <returns></returns>
    IMetricFunc SparseCategoricalCrossentropy(string name = "sparse_categorical_crossentropy",
        TF_DataType dtype = TF_DataType.TF_FLOAT,
        bool from_logits = false,
        int? ignore_class = null,
        Axis? axis = null);

    /// <summary>
    /// Computes the crossentropy metric between the labels and predictions.
    /// </summary>
    /// <returns></returns>
    IMetricFunc CategoricalAccuracy(string name = "categorical_accuracy", 
        TF_DataType dtype = TF_DataType.TF_FLOAT);

    /// <summary>
    /// Calculates how often predictions match integer labels.
    /// </summary>
    /// <returns></returns>
    IMetricFunc SparseCategoricalAccuracy(string name = "sparse_categorical_accuracy",
        TF_DataType dtype = TF_DataType.TF_FLOAT);

    /// <summary>
    /// Computes the cosine similarity between the labels and predictions.
    /// </summary>
    /// <returns></returns>
    IMetricFunc CosineSimilarity(string name = "cosine_similarity",
        TF_DataType dtype = TF_DataType.TF_FLOAT,
        Axis? axis = null);

    /// <summary>
    /// Computes F-1 Score.
    /// </summary>
    /// <returns></returns>
    IMetricFunc F1Score(int num_classes,
        string? average = null,
        float? threshold = null,
        string name = "f1_score",
        TF_DataType dtype = TF_DataType.TF_FLOAT);

    /// <summary>
    /// Computes F-Beta score.
    /// </summary>
    /// <returns></returns>
    IMetricFunc FBetaScore(int num_classes, 
        string? average = null,
        float beta = 0.1f,
        float? threshold = null,
        string name = "fbeta_score",
        TF_DataType dtype = TF_DataType.TF_FLOAT);

    /// <summary>
    /// Computes hamming loss.
    /// </summary>
    /// <param name="mode">multiclass or multilabel</param>
    /// <param name="threshold"></param>
    /// <param name="name"></param>
    /// <param name="dtype"></param>
    /// <returns></returns>
    IMetricFunc HammingLoss(string mode,
        float? threshold = null,
        string name = "hamming_loss",
        TF_DataType dtype = TF_DataType.TF_FLOAT);

    /// <summary>
    /// Computes how often targets are in the top K predictions.
    /// </summary>
    /// <param name="k"></param>
    /// <returns></returns>
    IMetricFunc TopKCategoricalAccuracy(int k = 5, 
        string name = "top_k_categorical_accuracy", 
        TF_DataType dtype = TF_DataType.TF_FLOAT);

    /// <summary>
    /// Computes how often integer targets are in the top K predictions.
    /// </summary>
    /// <param name="k"></param>
    /// <returns></returns>
    IMetricFunc SparseTopKCategoricalAccuracy(int k = 5,
        string name = "sparse_top_k_categorical_accuracy",
        TF_DataType dtype = TF_DataType.TF_FLOAT);

    /// <summary>
    /// Computes the precision of the predictions with respect to the labels.
    /// </summary>
    /// <param name="thresholds"></param>
    /// <param name="top_k"></param>
    /// <param name="class_id"></param>
    /// <param name="name"></param>
    /// <param name="dtype"></param>
    /// <returns></returns>
    IMetricFunc Precision(float thresholds = 0.5f, 
        int top_k = 0, 
        int class_id = 0, 
        string name = "recall", 
        TF_DataType dtype = TF_DataType.TF_FLOAT);

    /// <summary>
    /// Computes the recall of the predictions with respect to the labels.
    /// </summary>
    /// <param name="thresholds"></param>
    /// <param name="top_k"></param>
    /// <param name="class_id"></param>
    /// <param name="name"></param>
    /// <param name="dtype"></param>
    /// <returns></returns>
    IMetricFunc Recall(float thresholds = 0.5f, 
        int top_k = 0, 
        int class_id = 0, 
        string name = "recall", 
        TF_DataType dtype = TF_DataType.TF_FLOAT);
}
