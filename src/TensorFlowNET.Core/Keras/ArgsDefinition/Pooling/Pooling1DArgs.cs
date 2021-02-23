namespace Tensorflow.Keras.ArgsDefinition
{
    public class Pooling1DArgs : LayerArgs
    {
        /// <summary>
        /// The pooling function to apply, e.g. `tf.nn.max_pool2d`.
        /// </summary>
        public IPoolFunction PoolFunction { get; set; }

        /// <summary>
        /// specifying the size of the pooling window.
        /// </summary>
        public int PoolSize { get; set; }

        /// <summary>
        /// specifying the strides of the pooling operation.
        /// </summary>
        public int Strides { 
            get { return _strides.HasValue ? _strides.Value : PoolSize; }
            set { _strides = value; } 
        }
        private int? _strides = null;

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
