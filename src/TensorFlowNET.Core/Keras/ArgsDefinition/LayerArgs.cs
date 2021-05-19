namespace Tensorflow.Keras.ArgsDefinition
{
    public class LayerArgs
    {
        /// <summary>
        /// Indicates whether the layer's weights are updated during training
        /// and whether the layer's updates are run during training.
        /// </summary>
        public bool Trainable { get; set; } = true;

        public string Name { get; set; }

        /// <summary>
        /// Only applicable to input layers.
        /// </summary>
        public TF_DataType DType { get; set; } = TF_DataType.TF_FLOAT;

        /// <summary>
        /// Whether the `call` method can be used to build a TF graph without issues.
        /// This attribute has no effect if the model is created using the Functional
        /// API. Instead, `model.dynamic` is determined based on the internal layers.
        /// </summary>
        public bool Dynamic { get; set; } = false;

        /// <summary>
        /// Only applicable to input layers.
        /// </summary>
        public TensorShape InputShape { get; set; }

        /// <summary>
        /// Only applicable to input layers.
        /// </summary>
        public TensorShape BatchInputShape { get; set; }

        public int BatchSize { get; set; } = -1;

        /// <summary>
        /// Initial weight values.
        /// </summary>
        public float[] Weights { get; set; }

        /// <summary>
        /// Regularizer function applied to the output of the layer(its "activation").
        /// </summary>
        public IRegularizer ActivityRegularizer { get; set; }

        public bool Autocast { get; set; }

        public bool IsFromConfig { get; set; }
    }
}
