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
}
