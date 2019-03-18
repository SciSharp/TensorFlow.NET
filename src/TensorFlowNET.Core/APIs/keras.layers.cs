using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;

namespace Tensorflow
{
    public static partial class keras
    {
        public static class layers
        {
            public static Embedding Embedding(int input_dim, int output_dim,
            string embeddings_initializer = "uniform",
            bool mask_zero = false) => new Embedding(input_dim, output_dim,
                embeddings_initializer,
                mask_zero);
        }
    }
}
