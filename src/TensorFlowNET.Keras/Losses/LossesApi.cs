namespace Tensorflow.Keras.Losses
{
    public class LossesApi
    {
        public ILossFunc SparseCategoricalCrossentropy(bool from_logits = false)
            => new SparseCategoricalCrossentropy(from_logits: from_logits);

        public ILossFunc CategoricalCrossentropy(bool from_logits = false)
            => new CategoricalCrossentropy(from_logits: from_logits);
        
        public ILossFunc MeanSquaredError(string reduction = null)
          => new MeanSquaredError(reduction: reduction);
    }
}
