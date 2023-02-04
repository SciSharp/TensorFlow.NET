using Newtonsoft.Json;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class DropoutArgs : AutoSerializeLayerArgs
    {
        /// <summary>
        /// Float between 0 and 1. Fraction of the input units to drop.
        /// </summary>
        [JsonProperty("rate")]
        public float Rate { get; set; }

        /// <summary>
        /// 1D integer tensor representing the shape of the
        /// binary dropout mask that will be multiplied with the input.
        /// </summary>
        [JsonProperty("noise_shape")]
        public Shape NoiseShape { get; set; }

        /// <summary>
        /// random seed.
        /// </summary>
        [JsonProperty("seed")]
        public int? Seed { get; set; }

        public bool SupportsMasking { get; set; }
    }
}
