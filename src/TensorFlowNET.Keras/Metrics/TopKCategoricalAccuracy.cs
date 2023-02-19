namespace Tensorflow.Keras.Metrics;

public class TopKCategoricalAccuracy : MeanMetricWrapper
{
    public TopKCategoricalAccuracy(int k = 5, string name = "top_k_categorical_accuracy", TF_DataType dtype = TF_DataType.TF_FLOAT) 
        : base((yt, yp) => metrics_utils.sparse_top_k_categorical_matches(
                tf.math.argmax(yt, axis: -1), yp, k), 
            name: name, 
            dtype: dtype)
    {
    }
}
