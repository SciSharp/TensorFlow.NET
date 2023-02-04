using Newtonsoft.Json;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class EmbeddingArgs : AutoSerializeLayerArgs
    {
        [JsonProperty("input_dim")]
        public int InputDim { get; set; }
        [JsonProperty("output_dim")]
        public int OutputDim { get; set; }
        [JsonProperty("mask_zero")]
        public bool MaskZero { get; set; }
        [JsonProperty("input_length")]
        public int InputLength { get; set; } = -1;
        [JsonProperty("embeddings_initializer")]
        public IInitializer EmbeddingsInitializer { get; set; }
        [JsonProperty("activity_regularizer")]
        public override IRegularizer ActivityRegularizer { get => base.ActivityRegularizer; set => base.ActivityRegularizer = value; }

        // TODO: `embeddings_regularizer`, `embeddings_constraint`.
    }
}
