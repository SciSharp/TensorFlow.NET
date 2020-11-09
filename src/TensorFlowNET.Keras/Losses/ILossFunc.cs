namespace Tensorflow.Keras.Losses
{
    public interface ILossFunc
    {
        string Reduction { get; }
        Tensor Call(Tensor y_true, Tensor y_pred);
    }
}
