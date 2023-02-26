namespace Tensorflow.Keras.Losses
{
    public class LossesApi : ILossesApi
    {
        public ILossFunc BinaryCrossentropy(bool from_logits = false,
                float label_smoothing = 0,
                int axis = -1,
                string reduction = "auto",
                string name = "binary_crossentropy")
            => new BinaryCrossentropy(from_logits: from_logits,
                label_smoothing: label_smoothing,
                reduction: reduction,
                name: name);

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

        public ILossFunc CosineSimilarity(string reduction = null, int axis = -1, string name = null)
            => new CosineSimilarity(reduction: reduction, axis: axis, name: name);

        public ILossFunc Huber(string reduction = null, string name = null, Tensor delta=null)
            => new Huber(reduction: reduction, name: name, delta: delta);

        public ILossFunc LogCosh(string reduction = null, string name = null)
            => new LogCosh(reduction: reduction, name: name);

        public ILossFunc SigmoidFocalCrossEntropy(bool from_logits = false, 
                float alpha = 0.25F, 
                float gamma = 2, 
                string reduction = "none", 
                string name = "sigmoid_focal_crossentropy")
            => new SigmoidFocalCrossEntropy(from_logits: from_logits, 
                alpha: alpha, 
                gamma: gamma, 
                reduction: reduction, 
                name: name);
    }
}
