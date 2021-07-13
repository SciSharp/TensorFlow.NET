namespace Tensorflow.Keras.ArgsDefinition
{
    public class Pooling2DArgs : LayerArgs
    {
        /// <summary>
        /// The pooling function to apply, e.g. `tf.nn.max_pool2d`.
        /// </summary>
        public IPoolFunction PoolFunction { get; set; }

        /// <summary>
        /// specifying the size of the pooling window.
        /// </summary>
        public Shape PoolSize { get; set; }

        /// <summary>
        /// specifying the strides of the pooling operation.
        /// </summary>
        public Shape Strides { get; set; }

        /// <summary>
        /// The padding method, either 'valid' or 'same'.
        /// </summary>
        public string Padding { get; set; } = "valid";

        /// <summary>
        /// one of `channels_last` (default) or `channels_first`.
        /// </summary>
        public string DataFormat { get; set; }
    }
}
