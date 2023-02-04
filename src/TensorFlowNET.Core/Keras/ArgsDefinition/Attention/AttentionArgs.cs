using Newtonsoft.Json;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class AttentionArgs : BaseDenseAttentionArgs
    {

        /// <summary>
        /// If `true`, will create a scalar variable to scale the attention scores.
        /// </summary>
        [JsonProperty("use_scale")]
        public bool use_scale { get; set; } = false;

        /// <summary>
        /// Function to use to compute attention scores, one of
        /// `{"dot", "concat"}`. `"dot"` refers to the dot product between the query
        /// and key vectors. `"concat"` refers to the hyperbolic tangent of the
        /// concatenation of the query and key vectors.
        /// </summary>
        [JsonProperty("score_mode")]
        public string score_mode { get; set; } = "dot";

    }
}