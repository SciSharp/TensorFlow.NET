namespace Tensorflow.Keras.Metrics;

public class SparseTopKCategoricalAccuracy : MeanMetricWrapper
{
    public SparseTopKCategoricalAccuracy(int k = 5, string name = "sparse_top_k_categorical_accuracy", TF_DataType dtype = TF_DataType.TF_FLOAT) 
        : base((yt, yp) => metrics_utils.sparse_top_k_categorical_matches(yt, yp, k), 
            name: name, 
            dtype: dtype)
    {
    }
}
