namespace Tensorflow.Keras.Metrics;

public class Accuracy : MeanMetricWrapper
{
    public Accuracy(string name = "accuracy", TF_DataType dtype = TF_DataType.TF_FLOAT) 
        : base((yt, yp) => metrics_utils.accuracy(yt, yp), 
            name: name, 
            dtype: dtype)
    {
    }
}
