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

        public IMetricFunc TopKCategoricalAccuracy(int k = 5, string name = "top_k_categorical_accuracy", TF_DataType dtype = TF_DataType.TF_FLOAT)
            => new TopKCategoricalAccuracy(k: k, name: name, dtype: dtype);
    }
}
