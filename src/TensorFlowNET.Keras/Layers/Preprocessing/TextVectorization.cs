using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;

namespace Tensorflow.Keras.Layers
{
    public class TextVectorization : CombinerPreprocessingLayer
    {
        TextVectorizationArgs args;

        public TextVectorization(TextVectorizationArgs args)
            : base(args)
        {
            args.DType = TF_DataType.TF_STRING;
            // string standardize = "lower_and_strip_punctuation",
        }
    }
}
