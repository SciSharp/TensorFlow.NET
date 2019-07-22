namespace Tensorflow.Operations.Activation
{
    public interface IActivation
    {
        Tensor Activate(Tensor features, string name = null);
    }
}
