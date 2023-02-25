namespace Tensorflow.Keras.Metrics;

public class HammingLoss : MeanMetricWrapper
{
    public HammingLoss(string mode, 
        NDArray threshold = null,
        string name = "hamming_loss",
        TF_DataType dtype = TF_DataType.TF_FLOAT)
        : base((yt, yp) => metrics_utils.hamming_loss_fn(yt, yp, threshold, mode), 
            name: name, 
            dtype: dtype)
    {
        _dtype = dtype;
    }
}
