using System;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Preprocessings;

namespace Tensorflow.Keras
{
    public partial class Preprocessing
    {
        public Sequence sequence => new Sequence();
        public DatasetUtils dataset_utils => new DatasetUtils();

        public TextApi text => _text;

        private static TextApi _text = new TextApi();

        public TextVectorization TextVectorization(Func<Tensor, Tensor> standardize = null,
            string split = "whitespace",
            int max_tokens = -1,
            string output_mode = "int",
            int output_sequence_length = -1) => new TextVectorization(new TextVectorizationArgs
            {
                Standardize = standardize,
                Split = split,
                MaxTokens = max_tokens,
                OutputMode = output_mode,
                OutputSequenceLength = output_sequence_length
            });
    }
}
