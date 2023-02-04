using Newtonsoft.Json;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class LayerNormalizationArgs : AutoSerializeLayerArgs
    {
        [JsonProperty("axis")]
        public Axis Axis { get; set; } = -1;
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
        [JsonProperty("beta_regularizer")]
        public IRegularizer BetaRegularizer { get; set; }
        [JsonProperty("gamma_regularizer")]
        public IRegularizer GammaRegularizer { get; set; }

        // TODO: `beta_constraint` and `gamma_constraint`.
    }
}
