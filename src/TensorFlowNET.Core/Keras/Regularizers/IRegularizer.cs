namespace Tensorflow.Keras
{
    public interface IRegularizer
    {
        Tensor Apply(RegularizerArgs args);
    }
}
