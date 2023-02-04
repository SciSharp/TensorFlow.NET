using Newtonsoft.Json;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class BaseDenseAttentionArgs : AutoSerializeLayerArgs
    {

        /// <summary>
        /// Boolean. Set to `true` for decoder self-attention. Adds a mask such
        /// that position `i` cannot attend to positions `j > i`. This prevents the
        /// flow of information from the future towards the past.
        /// </summary>
        public bool causal { get; set; } = false;

        /// <summary>
        /// Float between 0 and 1. Fraction of the units to drop for the
        /// attention scores.
        /// </summary>
        [JsonProperty("dropout")]
        public float dropout { get; set; } = 0f;

    }
}