namespace Tensorflow.Keras.Losses
{
    public class LossesApi
    {
        public ILossFunc SparseCategoricalCrossentropy(bool from_logits = false)
            => new SparseCategoricalCrossentropy(from_logits: from_logits);
    }
}
