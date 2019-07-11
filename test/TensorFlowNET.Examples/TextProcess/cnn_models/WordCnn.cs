/*****************************************************************************
   Copyright 2018 The TensorFlow.NET Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow;
using static Tensorflow.Python;

namespace TensorFlowNET.Examples.Text
{
    public class WordCnn : ITextModel
    {
        public WordCnn(int vocabulary_size, int document_max_len, int num_class)
        {
            var embedding_size = 128;
            var learning_rate = 0.001f;
            var filter_sizes = new int[3, 4, 5];
            var num_filters = 100;

            var x = tf.placeholder(tf.int32, new TensorShape(-1, document_max_len), name: "x");
            var y = tf.placeholder(tf.int32, new TensorShape(-1), name: "y");
            var is_training = tf.placeholder(tf.@bool, new TensorShape(), name: "is_training");
            var global_step = tf.Variable(0, trainable: false);
            var keep_prob = tf.where(is_training, 0.5f, 1.0f);
            Tensor x_emb = null;

            with(tf.name_scope("embedding"), scope =>
            {
                var init_embeddings = tf.random_uniform(new int[] { vocabulary_size, embedding_size });
                var embeddings = tf.get_variable("embeddings", initializer: init_embeddings);
                x_emb = tf.nn.embedding_lookup(embeddings, x);
                x_emb = tf.expand_dims(x_emb, -1);
            });

            var pooled_outputs = new List<Tensor>();
            for (int len = 0; len < filter_sizes.Rank; len++)
            {
                int filter_size = filter_sizes.GetLength(len);
                var conv = tf.layers.conv2d(
                    x_emb,
                    filters: num_filters,
                    kernel_size: new int[] { filter_size, embedding_size },
                    strides: new int[] { 1, 1 },
                    padding: "VALID",
                    activation: tf.nn.relu());

                var pool = tf.layers.max_pooling2d(
                    conv,
                    pool_size: new[] { document_max_len - filter_size + 1, 1 },
                    strides: new[] { 1, 1 },
                    padding: "VALID");

                pooled_outputs.Add(pool);
            }

            var h_pool = tf.concat(pooled_outputs, 3);
            var h_pool_flat = tf.reshape(h_pool, new TensorShape(-1, num_filters * filter_sizes.Rank));
            Tensor h_drop = null;
            with(tf.name_scope("dropout"), delegate
            {
                h_drop = tf.nn.dropout(h_pool_flat, keep_prob);
            });

            Tensor logits = null;
            Tensor predictions = null;
            with(tf.name_scope("output"), delegate
            {
                logits = tf.layers.dense(h_drop, num_class);
                predictions = tf.argmax(logits, -1, output_type: tf.int32);
            });

            with(tf.name_scope("loss"), delegate
            {
                var sscel = tf.nn.sparse_softmax_cross_entropy_with_logits(logits: logits, labels: y);
                var loss = tf.reduce_mean(sscel);
                var adam = tf.train.AdamOptimizer(learning_rate);
                var optimizer = adam.minimize(loss, global_step: global_step);
            });

            with(tf.name_scope("accuracy"), delegate
            {
                var correct_predictions = tf.equal(predictions, y);
                var accuracy = tf.reduce_mean(tf.cast(correct_predictions, TF_DataType.TF_FLOAT), name: "accuracy");
            });
        }
    }
}
