using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow;
using TensorFlowNET.Examples.Text.cnn_models;
using static Tensorflow.Python;

namespace TensorFlowNET.Examples.TextClassification
{
    public class VdCnn : ITextClassificationModel
    {
        private int embedding_size;
        private int[] filter_sizes;
        private int[] num_filters;
        private int[] num_blocks;
        private float learning_rate;
        private IInitializer cnn_initializer;
        private IInitializer fc_initializer;
        public Tensor x { get; private set; }
        public Tensor y { get; private set; }
        public Tensor is_training { get; private set; }
        private RefVariable global_step;
        private RefVariable embeddings;
        private Tensor x_emb;
        private Tensor x_expanded;
        private Tensor logits;
        private Tensor predictions;
        private Tensor loss;

        public VdCnn(int alphabet_size, int document_max_len, int num_class)
        {
            embedding_size = 16;
            filter_sizes = new int[] { 3, 3, 3, 3, 3 };
            num_filters = new int[] { 64, 64, 128, 256, 512 };
            num_blocks = new int[] { 2, 2, 2, 2 };
            learning_rate = 0.001f;
            cnn_initializer = tf.keras.initializers.he_normal();
            fc_initializer = tf.truncated_normal_initializer(stddev: 0.05f);

            x = tf.placeholder(tf.int32, new TensorShape(-1, document_max_len), name: "x");
            y = tf.placeholder(tf.int32, new TensorShape(-1), name: "y");
            is_training = tf.placeholder(tf.@bool, new TensorShape(), name: "is_training");
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
            Tensor conv2 = null;
            Tensor conv3 = null;
            Tensor conv4 = null;
            Tensor h_flat = null;
            Tensor fc1_out = null;
            Tensor fc2_out = null;

            // First Convolution Layer
            with(tf.variable_scope("conv-0"), delegate
            {
                conv0 = tf.layers.conv2d(x_expanded,
                    filters: num_filters[0],
                    kernel_size: new int[] { filter_sizes[0], embedding_size },
                    kernel_initializer: cnn_initializer,
                    activation: tf.nn.relu());

                conv0 = tf.transpose(conv0, new int[] { 0, 1, 3, 2 });
            });

            with(tf.name_scope("conv-block-1"), delegate {
                conv1 = conv_block(conv0, 1);
            });

            with(tf.name_scope("conv-block-2"), delegate {
                conv2 = conv_block(conv1, 2);
            });

            with(tf.name_scope("conv-block-3"), delegate {
                conv3 = conv_block(conv2, 3);
            });

            with(tf.name_scope("conv-block-4"), delegate
            {
                conv4 = conv_block(conv3, 4, max_pool: false);
            });

            // ============= k-max Pooling =============
            with(tf.name_scope("k-max-pooling"), delegate
            {
                var h = tf.transpose(tf.squeeze(conv4, new int[] { -1 }), new int[] { 0, 2, 1 });
                var top_k = tf.nn.top_k(h, k: 8, sorted: false)[0];
                h_flat = tf.reshape(top_k, new int[] { -1, 512 * 8 });
            });

            // ============= Fully Connected Layers =============
            with(tf.name_scope("fc-1"), scope =>
            {
                fc1_out = tf.layers.dense(h_flat, 2048, activation: tf.nn.relu(), kernel_initializer: fc_initializer);
            });

            with(tf.name_scope("fc-2"), scope =>
            {
                fc2_out = tf.layers.dense(fc1_out, 2048, activation: tf.nn.relu(), kernel_initializer: fc_initializer);
            });

            with(tf.name_scope("fc-3"), scope =>
            {
                logits = tf.layers.dense(fc2_out, num_class, activation: null, kernel_initializer: fc_initializer);
                predictions = tf.argmax(logits, -1, output_type: tf.int32);
            });

            // ============= Loss and Accuracy =============
            with(tf.name_scope("loss"), delegate
            {
                var y_one_hot = tf.one_hot(y, num_class);
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits: logits, labels: y_one_hot));

                var update_ops = tf.get_collection(ops.GraphKeys.UPDATE_OPS) as List<object>;
                with(tf.control_dependencies(update_ops.Select(x => (Operation)x).ToArray()), delegate
                {
                    var adam = tf.train.AdamOptimizer(learning_rate);
                    adam.minimize(loss, global_step: global_step);
                });
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
                        conv = tf.nn.relu(conv);
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
