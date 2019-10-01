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

using NumSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// How to optimise your input pipeline with queues and multi-threading
    /// https://blog.metaflow.fr/tensorflow-how-to-optimise-your-input-pipeline-with-queues-and-multi-threading-e7c3874157e0
    /// </summary>
    public class FullyConnected : IExample
    {
        public bool Enabled { get; set; } = true;
        public bool IsImportingGraph { get; set; }

        public string Name => "Fully Connected Neural Network";

        Tensor input = null;
        Tensor x_inputs_data = null;
        Tensor y_inputs_data = null;
        Tensor accuracy = null;
        Tensor y_true = null;
        Tensor loss_op = null;
        Operation train_op = null;

        public Graph BuildGraph()
        {
            var g = tf.get_default_graph();
            
            Tensor z = null;
            
            tf_with(tf.variable_scope("placeholder"), delegate
            {
                input = tf.placeholder(tf.float32, shape: (-1, 1024));
                y_true = tf.placeholder(tf.int32, shape: (-1, 1));
            });

            tf_with(tf.variable_scope("FullyConnected"), delegate
            {
                var w = tf.get_variable("w", shape: (1024, 1024), initializer: tf.random_normal_initializer(stddev: 0.1f));
                var b = tf.get_variable("b", shape: 1024, initializer: tf.constant_initializer(0.1));
                z = tf.matmul(input, w) + b;
                var y = tf.nn.relu(z);

                var w2 = tf.get_variable("w2", shape: (1024, 1), initializer: tf.random_normal_initializer(stddev: 0.1f));
                var b2 = tf.get_variable("b2", shape: 1, initializer: tf.constant_initializer(0.1));
                z = tf.matmul(y, w2) + b2;
            });

            tf_with(tf.variable_scope("Loss"), delegate
            {
                var losses = tf.nn.sigmoid_cross_entropy_with_logits(tf.cast(y_true, tf.float32), z);
                loss_op = tf.reduce_mean(losses);
            });

            tf_with(tf.variable_scope("Accuracy"), delegate
            {
                var y_pred = tf.cast(z > 0, tf.int32);
                accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_true), tf.float32));
                // accuracy = tf.Print(accuracy, data =[accuracy], message = "accuracy:")
            });

            // We add the training operation, ...
            var adam = tf.train.AdamOptimizer(0.01f);
            train_op = adam.minimize(loss_op, name: "train_op");

            return g;
        }

        public Graph ImportGraph()
        {
            throw new NotImplementedException();
        }

        public void Predict(Session sess)
        {
            throw new NotImplementedException();
        }

        public void PrepareData()
        {
            // batches of 128 samples, each containing 1024 data points
            x_inputs_data = tf.random_normal(new[] { 128, 1024 }, mean: 0, stddev: 1);
            // We will try to predict this law:
            // predict 1 if the sum of the elements is positive and 0 otherwise
            y_inputs_data = tf.cast(tf.reduce_sum(x_inputs_data, axis: 1, keepdims: true) > 0, tf.int32);
        }

        public bool Run()
        {
            PrepareData();
            var g = BuildGraph();
            using (var sess = tf.Session())
                Train(sess);
            return true;
        }

        public void Test(Session sess)
        {
            throw new NotImplementedException();
        }

        public void Train(Session sess)
        {
            var sw = new Stopwatch();
            sw.Start();
            // init variables
            sess.run(tf.global_variables_initializer());

            // check the accuracy before training
            var (x_input, y_input) = sess.run((x_inputs_data, y_inputs_data));
            sess.run(accuracy, (input, x_input), (y_true, y_input));

            // training
            foreach (var i in range(5000))
            {
                // by sampling some input data (fetching)
                (x_input, y_input) = sess.run((x_inputs_data, y_inputs_data));
                var (_, loss) = sess.run((train_op, loss_op), (input, x_input), (y_true, y_input));

                // We regularly check the loss
                if (i % 500 == 0)
                    print($"iter:{i} - loss:{loss}");
            }

            // Finally, we check our final accuracy
            (x_input, y_input) = sess.run((x_inputs_data, y_inputs_data));
            sess.run(accuracy, (input, x_input), (y_true, y_input));

            print($"Time taken: {sw.Elapsed.TotalSeconds}s");
        }
    }
}
