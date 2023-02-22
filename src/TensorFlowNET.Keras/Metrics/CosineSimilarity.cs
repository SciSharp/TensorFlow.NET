namespace Tensorflow.Keras.Metrics;

public class CosineSimilarity : MeanMetricWrapper
{
    public CosineSimilarity(string name = "cosine_similarity", TF_DataType dtype = TF_DataType.TF_FLOAT, Axis? axis = null) 
        : base((yt, yp) => metrics_utils.cosine_similarity(yt, yp, axis: axis ?? -1), 
            name: name, 
            dtype: dtype)
    {
    }
}
