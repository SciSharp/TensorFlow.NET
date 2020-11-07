namespace Tensorflow.Keras.Losses
{
    public class LossFunctionWrapper : Loss
    {
        public LossFunctionWrapper(string reduction = ReductionV2.AUTO,
            string name = null)
            : base(reduction: reduction,
                  name: name)
        {
        }
    }
}
