using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Layers
{
    public class Embedding : Layer
    {
        private int input_dim;
        private int output_dim;
        private bool mask_zero;

        public Embedding(int input_dim, int output_dim,
            IInitializer embeddings_initializer = null,
            bool mask_zero = false)
        {
            this.input_dim = input_dim;
            this.output_dim = output_dim;
            if (embeddings_initializer == null)
                embeddings_initializer = tf.uniform_initializer;
            this.mask_zero = mask_zero;
            supports_masking = mask_zero;
        }
    }
}
