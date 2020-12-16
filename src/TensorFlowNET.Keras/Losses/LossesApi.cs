namespace Tensorflow.Keras.Losses
{
    public class LossesApi
    {
        public ILossFunc SparseCategoricalCrossentropy(string reduction = null, string name = null,bool from_logits = false)
            => new SparseCategoricalCrossentropy(reduction: reduction, name: name,from_logits: from_logits);

        public ILossFunc CategoricalCrossentropy(string reduction = null, string name = null,bool from_logits = false)
            => new CategoricalCrossentropy(reduction: reduction, name: name,from_logits: from_logits);
        
        public ILossFunc MeanSquaredError(string reduction = null, string name = null)
            => new MeanSquaredError(reduction: reduction, name:name);
        public ILossFunc MeanSquaredLogarithmicError(string reduction = null, string name = null)
           => new MeanSquaredLogarithmicError(reduction: reduction, name: name);

        public ILossFunc MeanAbsolutePercentageError(string reduction = null, string name = null)
           => new MeanAbsolutePercentageError(reduction: reduction, name: name);

        public ILossFunc MeanAbsoluteError(string reduction = null, string name = null)
           => new MeanAbsoluteError(reduction: reduction, name: name);

        public ILossFunc CosineSimilarity(string reduction = null, string name = null,int axis=-1)
            => new CosineSimilarity(reduction: reduction, name: name, axis: axis);

        public ILossFunc Huber(string reduction = null, string name = null, Tensor delta=null)
            => new Huber(reduction: reduction, name: name, delta: delta);

        public ILossFunc LogCosh(string reduction = null, string name = null)
            => new LogCosh(reduction: reduction, name: name);

    }
}
