using Newtonsoft.Json;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class BatchNormalizationArgs : AutoSerializeLayerArgs
    {
        [JsonProperty("axis")]
        public Shape Axis { get; set; } = -1;
        [JsonProperty("momentum")]
        public float Momentum { get; set; } = 0.99f;
        [JsonProperty("epsilon")]
        public float Epsilon { get; set; } = 1e-3f;
        [JsonProperty("center")]
        public bool Center { get; set; } = true;
        [JsonProperty("scale")]
        public bool Scale { get; set; } = true;
        [JsonProperty("beta_initializer")]
        public IInitializer BetaInitializer { get; set; } = tf.zeros_initializer;
        [JsonProperty("gamma_initializer")]
        public IInitializer GammaInitializer { get; set; } = tf.ones_initializer;
        [JsonProperty("moving_mean_initializer")]
        public IInitializer MovingMeanInitializer { get; set; } = tf.zeros_initializer;
        [JsonProperty("moving_variance_initializer")]
        public IInitializer MovingVarianceInitializer { get; set; } = tf.ones_initializer;
        [JsonProperty("beta_regularizer")]
        public IRegularizer BetaRegularizer { get; set; }
        [JsonProperty("gamma_regularizer")]
        public IRegularizer GammaRegularizer { get; set; }
        // TODO: `beta_constraint` and `gamma_constraint`.
        [JsonProperty("renorm")]
        public bool Renorm { get; set; }
        // TODO: `renorm_clipping` and `virtual_batch_size`.
        [JsonProperty("renorm_momentum")]
        public float RenormMomentum { get; set; } = 0.99f;
    }
}
