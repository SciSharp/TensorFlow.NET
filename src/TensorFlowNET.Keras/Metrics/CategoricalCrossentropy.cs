namespace Tensorflow.Keras.Metrics;

public class CategoricalCrossentropy : MeanMetricWrapper
{
    public CategoricalCrossentropy(string name = "categorical_crossentropy",
        TF_DataType dtype = TF_DataType.TF_FLOAT,
        bool from_logits = false,
        float label_smoothing = 0f,
        Axis? axis = null)
        : base((yt, yp) => keras.metrics.categorical_crossentropy(
                yt, yp, from_logits: from_logits, label_smoothing: label_smoothing, axis: axis ?? -1), 
            name: name, 
            dtype: dtype)
    {
    }
}
