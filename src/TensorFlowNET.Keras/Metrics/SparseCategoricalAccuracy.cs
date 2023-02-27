namespace Tensorflow.Keras.Metrics;

public class SparseCategoricalAccuracy : MeanMetricWrapper
{
    public SparseCategoricalAccuracy(string name = "sparse_categorical_accuracy", TF_DataType dtype = TF_DataType.TF_FLOAT) 
        : base((yt, yp) => metrics_utils.sparse_categorical_matches(yt, yp), 
            name: name, 
            dtype: dtype)
    {
    }
}
