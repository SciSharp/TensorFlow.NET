namespace Tensorflow.Keras.Metrics;

public class CategoricalAccuracy : MeanMetricWrapper
{
    public CategoricalAccuracy(string name = "categorical_accuracy", TF_DataType dtype = TF_DataType.TF_FLOAT) 
        : base((yt, yp) => metrics_utils.sparse_categorical_matches(
                tf.math.argmax(yt, axis: -1), yp), 
            name: name, 
            dtype: dtype)
    {
    }
}
