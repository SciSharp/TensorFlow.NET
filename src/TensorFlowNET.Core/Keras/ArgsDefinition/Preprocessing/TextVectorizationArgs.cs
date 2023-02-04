using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class TextVectorizationArgs : PreprocessingLayerArgs
    {
        [JsonProperty("standardize")]
        public Func<Tensor, Tensor> Standardize { get; set; }
        [JsonProperty("split")]
        public string Split { get; set; } = "standardize";
        [JsonProperty("max_tokens")]
        public int MaxTokens { get; set; } = -1;
        [JsonProperty("output_mode")]
        public string OutputMode { get; set; } = "int";
        [JsonProperty("output_sequence_length")]
        public int OutputSequenceLength { get; set; } = -1;
        [JsonProperty("vocabulary")]
        public string[] Vocabulary { get; set; }

        // TODO: Add `ngrams`, `sparse`, `ragged`, `idf_weights`, `encoding`
    }
}
