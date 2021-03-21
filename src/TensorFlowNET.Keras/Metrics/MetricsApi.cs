using static Tensorflow.KerasApi;

namespace Tensorflow.Keras.Metrics
{
    public class MetricsApi
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
            var y_pred_rank = y_pred.TensorShape.ndim;
            var y_true_rank = y_true.TensorShape.ndim;
            // If the shape of y_true is (num_samples, 1), squeeze to (num_samples,)
            if (y_true_rank != -1 && y_pred_rank != -1
                && y_true.shape.Length == y_pred.shape.Length)
                y_true = array_ops.squeeze(y_true, axis: new[] { -1 });
            y_pred = math_ops.argmax(y_pred, -1);

            // If the predicted output and actual output types don't match, force cast them
            // to match.
            if (y_pred.dtype != y_true.dtype)
                y_pred = math_ops.cast(y_pred, y_true.dtype);

            return math_ops.cast(math_ops.equal(y_true, y_pred), TF_DataType.TF_FLOAT);
        }
    }
}
