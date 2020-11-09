namespace Tensorflow.Keras
{
    public class Regularizers
    {
        public IRegularizer l2(float l2 = 0.01f)
            => new L2(l2);
    }
}
