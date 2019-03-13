using System;
using System.Collections.Generic;
using System.Linq;
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
        private Tensor x_expanded;

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

            // Embedding Layer
            with(tf.name_scope("embedding"), delegate
            {
                var init_embeddings = tf.random_uniform(new int[] { alphabet_size, embedding_size }, -1.0f, 1.0f);
                embeddings = tf.get_variable("embeddings", initializer: init_embeddings);
                x_emb = tf.nn.embedding_lookup(embeddings, x);
                x_expanded = tf.expand_dims(x_emb, -1);
            });

            Tensor conv0 = null;
            Tensor conv1 = null;

            // First Convolution Layer
            with(tf.variable_scope("conv-0"), delegate
            {
                conv0 = tf.layers.conv2d(x_expanded,
                    filters: num_filters[0],
                    kernel_size: new int[] { filter_sizes[0], embedding_size },
                    kernel_initializer: cnn_initializer,
                    activation: tf.nn.relu);

                conv0 = tf.transpose(conv0, new int[] { 0, 1, 3, 2 });
            });

            with(tf.name_scope("conv-block-1"), delegate {
                conv1 = conv_block(conv0, 1);
            });
            
        }

        private Tensor conv_block(Tensor input, int i, bool max_pool = true)
        {
            return with(tf.variable_scope($"conv-block-{i}"), delegate
            {
                Tensor conv = null;
                // Two "conv-batch_norm-relu" layers.
                foreach (var j in Enumerable.Range(0, 2))
                {
                    with(tf.variable_scope($"conv-{j}"), delegate
                    {
                        // convolution
                        conv = tf.layers.conv2d(
                            input,
                            filters: num_filters[i],
                            kernel_size: new int[] { filter_sizes[i], num_filters[i - 1] },
                            kernel_initializer: cnn_initializer,
                            activation: null);
                        // batch normalization
                        conv = tf.layers.batch_normalization(conv, training: is_training);
                        // relu
                        conv = tf.nn.relu.Activate(conv);
                        conv = tf.transpose(conv, new int[] { 0, 1, 3, 2 });
                    });
                }
                
                if (max_pool)
                {
                    // Max pooling
                    return tf.layers.max_pooling2d(
                        conv,
                        pool_size: new int[] { 3, 1 },
                        strides: new int[] { 2, 1 },
                        padding: "SAME");
                }
                else
                {
                    return conv;
                }
            });
        }
    }
}
