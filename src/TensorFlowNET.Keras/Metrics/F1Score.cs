namespace Tensorflow.Keras.Metrics;

public class F1Score : FBetaScore
{
    public F1Score(int num_classes,
        string? average = null,
        float? threshold = null,
        string name = "f1_score",
        TF_DataType dtype = TF_DataType.TF_FLOAT)
        : base(num_classes, average: average, threshold: threshold, beta: 1f, name: name, dtype: dtype)
    {
    }
}
