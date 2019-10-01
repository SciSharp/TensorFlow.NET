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
    public class FullyConnectedInQueue : IExample
    {
        public bool Enabled { get; set; } = false;
        public bool IsImportingGraph { get; set; }

        public string Name => "Fully Connected Neural Network In Queue";

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

            // We build our small model: a basic two layers neural net with ReLU
            tf_with(tf.variable_scope("queue"), delegate
            {
                // enqueue 5 batches
                var q = tf.FIFOQueue(capacity: 5, dtype: tf.float32);
                // We use the "enqueue" operation so 1 element of the queue is the full batch
                var enqueue_op = q.enqueue(x_inputs_data);
                var numberOfThreads = 1;
                // var qr = tf.train.QueueRunner(q, [enqueue_op] * numberOfThreads);
            });

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
