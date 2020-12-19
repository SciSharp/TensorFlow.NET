namespace Tensorflow.Keras.Losses
{
    public interface ILossFunc
    {
        public string Reduction { get; }
        public string Name { get; }
        Tensor Call(Tensor y_true, Tensor y_pred, Tensor sample_weight = null);
    }
}
