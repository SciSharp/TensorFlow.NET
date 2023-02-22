namespace Tensorflow.Keras.Metrics;

public class BinaryAccuracy : MeanMetricWrapper
{
    public BinaryAccuracy(string name = "binary_accuracy", TF_DataType dtype = TF_DataType.TF_FLOAT, float threshold = 0.5f) 
        : base((yt, yp) => metrics_utils.binary_matches(yt, yp), 
            name: name, 
            dtype: dtype)
    {
    }
}
