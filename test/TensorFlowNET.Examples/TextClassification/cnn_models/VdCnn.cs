using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;

namespace TensorFlowNET.Examples.TextClassification
{
    public class VdCnn : Python
    {
        private int embedding_size;
        private int[] filter_sizes;
        private int[] num_filters;
        private int[] num_blocks;
        private float learning_rate;
        private IInitializer cnn_initializer;
        private Tensor x;
        private Tensor y;
        private Tensor is_training;
        private RefVariable global_step;
        private RefVariable embeddings;
        private Tensor x_emb;

        public VdCnn(int alphabet_size, int document_max_len, int num_class)
        {
            embedding_size = 16;
            filter_sizes = new int[] { 3, 3, 3, 3, 3 };
            num_filters = new int[] { 64, 64, 128, 256, 512 };
            num_blocks = new int[] { 2, 2, 2, 2 };
            learning_rate = 0.001f;
            cnn_initializer = tf.keras.initializers.he_normal();
            x = tf.placeholder(tf.int32, new TensorShape(-1, document_max_len), name: "x");
            y = tf.placeholder(tf.int32, new TensorShape(-1), name: "y");
            is_training = tf.placeholder(tf.boolean, new TensorShape(), name: "is_training");
            global_step = tf.Variable(0, trainable: false);

            with(tf.name_scope("embedding"), delegate
            {
                var init_embeddings = tf.random_uniform(new int[] { alphabet_size, embedding_size }, -1.0f, 1.0f);
                embeddings = tf.get_variable("embeddings", initializer: init_embeddings);
                // x_emb = tf.nn.embedding_lookup(embeddings, x);
            });
        }
    }
}
