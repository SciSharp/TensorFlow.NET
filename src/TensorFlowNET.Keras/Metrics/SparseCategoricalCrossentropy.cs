namespace Tensorflow.Keras.Metrics;

public class SparseCategoricalCrossentropy : MeanMetricWrapper
{
    public SparseCategoricalCrossentropy(string name = "sparse_categorical_crossentropy",
        TF_DataType dtype = TF_DataType.TF_FLOAT,
        bool from_logits = false,
        int? ignore_class = null,
        Axis? axis = null)
        : base((yt, yp) => keras.metrics.sparse_categorical_crossentropy(
                yt, yp, from_logits: from_logits, ignore_class: ignore_class, axis: axis ?? -1), 
            name: name, 
            dtype: dtype)
    {
    }
}
