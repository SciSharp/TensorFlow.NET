using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class TextVectorizationArgs : PreprocessingLayerArgs
    {
        public Func<Tensor, Tensor> Standardize { get; set; }
        public string Split { get; set; } = "standardize";
        public int MaxTokens { get; set; } = -1;
        public string OutputMode { get; set; } = "int";
        public int OutputSequenceLength { get; set; } = -1;
        public string[] Vocabulary { get; set; }
    }
}
