using Newtonsoft.Json;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class Pooling1DArgs : AutoSerializeLayerArgs
    {
        /// <summary>
        /// The pooling function to apply, e.g. `tf.nn.max_pool2d`.
        /// </summary>
        public IPoolFunction PoolFunction { get; set; }

        /// <summary>
        /// specifying the size of the pooling window.
        /// </summary>
        [JsonProperty("pool_size")]
        public int PoolSize { get; set; }

        /// <summary>
        /// specifying the strides of the pooling operation.
        /// </summary>
        [JsonProperty("strides")]
        public int Strides { 
            get { return _strides.HasValue ? _strides.Value : PoolSize; }
            set { _strides = value; } 
        }
        private int? _strides = null;

        /// <summary>
        /// The padding method, either 'valid' or 'same'.
        /// </summary>
        [JsonProperty("padding")]
        public string Padding { get; set; } = "valid";

        /// <summary>
        /// one of `channels_last` (default) or `channels_first`.
        /// </summary>
        [JsonProperty("data_format")]
        public string DataFormat { get; set; }
    }
}
