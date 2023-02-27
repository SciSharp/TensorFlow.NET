namespace Tensorflow.Keras.Metrics
{
    public class MetricsApi : IMetricsApi
    {
        public Tensor binary_accuracy(Tensor y_true, Tensor y_pred)
        {
            float threshold = 0.5f;
            y_pred = math_ops.cast(y_pred > threshold, y_pred.dtype);
            return keras.backend.mean(math_ops.equal(y_true, y_pred), axis: -1);
        }

        public Tensor categorical_accuracy(Tensor y_true, Tensor y_pred)
        {
            var eql = math_ops.equal(math_ops.argmax(y_true, -1), math_ops.argmax(y_pred, -1));
            return math_ops.cast(eql, TF_DataType.TF_FLOAT);
        }

        public Tensor categorical_crossentropy(Tensor y_true, Tensor y_pred, bool from_logits = false, float label_smoothing = 0, Axis? axis = null)
        {
            y_true = tf.cast(y_true, y_pred.dtype);
            // var label_smoothing_tensor = tf.convert_to_tensor(label_smoothing, dtype: y_pred.dtype);
            if (label_smoothing > 0)
            {
                var num_classes = tf.cast(tf.shape(y_true)[-1], y_pred.dtype);
                y_true = y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes);
            }
            return keras.backend.categorical_crossentropy(y_true, y_pred, from_logits: from_logits, axis: axis);
        }

        public Tensor sparse_categorical_crossentropy(Tensor y_true, Tensor y_pred, bool from_logits = false, int? ignore_class = null, Axis? axis = null)
        {
            return keras.backend.sparse_categorical_crossentropy(y_true, y_pred, from_logits: from_logits, axis: axis ?? -1, ignore_class: ignore_class);
        }

        /// <summary>
        /// Calculates how often predictions matches integer labels.
        /// </summary>
        /// <param name="y_true">Integer ground truth values.</param>
        /// <param name="y_pred">The prediction values.</param>
        /// <returns>Sparse categorical accuracy values.</returns>
        public Tensor sparse_categorical_accuracy(Tensor y_true, Tensor y_pred)
        {
            var y_pred_rank = y_pred.shape.ndim;
            var y_true_rank = y_true.shape.ndim;
            // If the shape of y_true is (num_samples, 1), squeeze to (num_samples,)
            if (y_true_rank != -1 && y_pred_rank != -1
                && y_true.shape.ndim == y_pred.shape.ndim)
                y_true = array_ops.squeeze(y_true, axis: new[] { -1 });
            y_pred = math_ops.argmax(y_pred, -1);

            // If the predicted output and actual output types don't match, force cast them
            // to match.
            if (y_pred.dtype != y_true.dtype)
                y_pred = math_ops.cast(y_pred, y_true.dtype);

            return math_ops.cast(math_ops.equal(y_true, y_pred), TF_DataType.TF_FLOAT);
        }

        public Tensor mean_absolute_error(Tensor y_true, Tensor y_pred)
        {
            y_true = math_ops.cast(y_true, y_pred.dtype);
            return keras.backend.mean(math_ops.abs(y_pred - y_true), axis: -1);
        }

        public Tensor mean_absolute_percentage_error(Tensor y_true, Tensor y_pred)
        {
            y_true = math_ops.cast(y_true, y_pred.dtype);
            var diff = (y_true - y_pred) / math_ops.maximum(math_ops.abs(y_true), keras.backend.epsilon());
            return 100f * keras.backend.mean(math_ops.abs(diff), axis: -1);
        }

        public Tensor top_k_categorical_accuracy(Tensor y_true, Tensor y_pred, int k = 5)
        {
            return metrics_utils.sparse_top_k_categorical_matches(
                tf.math.argmax(y_true, axis: -1), y_pred, k
            );
        }

        public IMetricFunc Accuracy(string name = "accuracy", TF_DataType dtype = TF_DataType.TF_FLOAT)
            => new Accuracy(name: name, dtype: dtype);

        public IMetricFunc BinaryAccuracy(string name = "binary_accuracy", TF_DataType dtype = TF_DataType.TF_FLOAT, float threshold = 5)
            => new BinaryAccuracy();

        public IMetricFunc CategoricalAccuracy(string name = "categorical_accuracy", TF_DataType dtype = TF_DataType.TF_FLOAT)
            => new CategoricalAccuracy(name: name, dtype: dtype);

        public IMetricFunc CategoricalCrossentropy(string name = "categorical_crossentropy", TF_DataType dtype = TF_DataType.TF_FLOAT, bool from_logits = false, float label_smoothing = 0, Axis? axis = null)
            => new CategoricalCrossentropy();

        public IMetricFunc CosineSimilarity(string name = "cosine_similarity", TF_DataType dtype = TF_DataType.TF_FLOAT, Axis? axis = null)
            => new CosineSimilarity(name: name, dtype: dtype, axis: axis ?? -1);

        public IMetricFunc F1Score(int num_classes, string? average = null, float? threshold = null, string name = "f1_score", TF_DataType dtype = TF_DataType.TF_FLOAT)
            => new F1Score(num_classes, average: average, threshold: threshold, name: name, dtype: dtype);

        public IMetricFunc FBetaScore(int num_classes, string? average = null, float beta = 0.1F, float? threshold = null, string name = "fbeta_score", TF_DataType dtype = TF_DataType.TF_FLOAT)
            => new FBetaScore(num_classes, average: average,beta: beta, threshold: threshold, name: name, dtype: dtype);

        public IMetricFunc HammingLoss(string mode, float? threshold = null, string name = "hamming_loss", TF_DataType dtype = TF_DataType.TF_FLOAT)
            => new HammingLoss(mode, threshold: threshold, name: name, dtype: dtype);

        public IMetricFunc TopKCategoricalAccuracy(int k = 5, string name = "top_k_categorical_accuracy", TF_DataType dtype = TF_DataType.TF_FLOAT)
            => new TopKCategoricalAccuracy(k: k, name: name, dtype: dtype);

        public IMetricFunc Precision(float thresholds = 0.5f, int top_k = 0, int class_id = 0, string name = "precision", TF_DataType dtype = TF_DataType.TF_FLOAT)
            => new Precision(thresholds: thresholds, top_k: top_k, class_id: class_id, name: name, dtype: dtype);

        public IMetricFunc Recall(float thresholds = 0.5f, int top_k = 0, int class_id = 0, string name = "recall", TF_DataType dtype = TF_DataType.TF_FLOAT)
            => new Recall(thresholds: thresholds, top_k: top_k, class_id: class_id, name: name, dtype: dtype);

        public IMetricFunc SparseCategoricalCrossentropy(string name = "sparse_categorical_crossentropy", TF_DataType dtype = TF_DataType.TF_FLOAT, bool from_logits = false, int? ignore_class = null, Axis? axis = null)
            => new SparseCategoricalCrossentropy(name: name, dtype: dtype, from_logits: from_logits, ignore_class: ignore_class, axis: axis ?? -1);

        public IMetricFunc SparseTopKCategoricalAccuracy(int k = 5, string name = "sparse_top_k_categorical_accuracy", TF_DataType dtype = TF_DataType.TF_FLOAT)
            => new SparseTopKCategoricalAccuracy(k: k, name: name, dtype: dtype);

        public IMetricFunc SparseCategoricalAccuracy(string name = "sparse_categorical_accuracy", TF_DataType dtype = TF_DataType.TF_FLOAT)
            => new SparseCategoricalAccuracy(name: name, dtype: dtype);
    }
}
