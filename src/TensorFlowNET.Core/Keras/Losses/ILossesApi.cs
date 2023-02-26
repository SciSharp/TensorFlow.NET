namespace Tensorflow.Keras.Losses;

public interface ILossesApi
{
    ILossFunc BinaryCrossentropy(bool from_logits = false, 
        float label_smoothing = 0f,
        int axis = -1,
        string reduction = "auto",
        string name = "binary_crossentropy");

    ILossFunc SparseCategoricalCrossentropy(string reduction = null, 
        string name = null, 
        bool from_logits = false);

    ILossFunc CategoricalCrossentropy(string reduction = null, 
        string name = null, 
        bool from_logits = false);

    ILossFunc MeanSquaredError(string reduction = null, 
        string name = null);

    ILossFunc MeanSquaredLogarithmicError(string reduction = null, 
        string name = null);

    ILossFunc MeanAbsolutePercentageError(string reduction = null, 
        string name = null);

    ILossFunc MeanAbsoluteError(string reduction = null, 
        string name = null);

    ILossFunc CosineSimilarity(string reduction = null,
        int axis = -1,
        string name = null);

    ILossFunc Huber(string reduction = null, 
        string name = null, 
        Tensor delta = null);

    ILossFunc LogCosh(string reduction = null, 
        string name = null);

    /// <summary>
    /// Implements the focal loss function.
    /// </summary>
    /// <param name="from_logits"></param>
    /// <param name="alpha"></param>
    /// <param name="gamma"></param>
    /// <param name="reduction"></param>
    /// <param name="name"></param>
    /// <returns></returns>
    ILossFunc SigmoidFocalCrossEntropy(bool from_logits = false,
        float alpha = 0.25f,
        float gamma = 2.0f,
        string reduction = "none",
        string name = "sigmoid_focal_crossentropy");
}
