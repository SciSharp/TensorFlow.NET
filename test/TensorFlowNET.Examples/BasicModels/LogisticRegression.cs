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
using System.Diagnostics;
using System.IO;
using Tensorflow;
using Tensorflow.Hub;
using static Tensorflow.Binding;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// A logistic regression learning algorithm example using TensorFlow library.
    /// This example is using the MNIST database of handwritten digits
    /// https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/logistic_regression.py
    /// </summary>
    public class LogisticRegression : IExample
    {
        public bool Enabled { get; set; } = true;
        public string Name => "Logistic Regression";
        public bool IsImportingGraph { get; set; } = false;


        public int training_epochs = 10;
        public int? train_size = null;
        public int validation_size = 5000;
        public int? test_size = null;
        public int batch_size = 100;

        private float learning_rate = 0.01f;
        private int display_step = 1;

        Datasets<MnistDataSet> mnist;

        public bool Run()
        {
            PrepareData();

            // tf Graph Input
            var x = tf.placeholder(tf.float32, new TensorShape(-1, 784)); // mnist data image of shape 28*28=784
            var y = tf.placeholder(tf.float32, new TensorShape(-1, 10)); // 0-9 digits recognition => 10 classes

            // Set model weights
            var W = tf.Variable(tf.zeros(new Shape(784, 10)));
            var b = tf.Variable(tf.zeros(new Shape(10)));

            // Construct model
            var pred = tf.nn.softmax(tf.matmul(x, W) + b); // Softmax

            // Minimize error using cross entropy
            var cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices: 1));

            // Gradient Descent
            var optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost);

            // Initialize the variables (i.e. assign their default value)
            var init = tf.global_variables_initializer();

            var sw = new Stopwatch();

            using (var sess = tf.Session())
            {
                // Run the initializer
                sess.run(init);

                // Training cycle
                foreach (var epoch in range(training_epochs))
                {
                    sw.Start();

                    var avg_cost = 0.0f;
                    var total_batch = mnist.Train.NumOfExamples / batch_size;
                    // Loop over all batches
                    foreach (var i in range(total_batch))
                    {
                        var (batch_xs, batch_ys) = mnist.Train.GetNextBatch(batch_size);
                        // Run optimization op (backprop) and cost op (to get loss value)
                        (_, float c) = sess.run((optimizer, cost), 
                            (x, batch_xs), 
                            (y, batch_ys));

                        // Compute average loss
                        avg_cost += c / total_batch;
                    }

                    sw.Stop();

                    // Display logs per epoch step
                    if ((epoch + 1) % display_step == 0)
                        print($"Epoch: {(epoch + 1):D4} Cost: {avg_cost:G9} Elapse: {sw.ElapsedMilliseconds}ms");

                    sw.Reset();
                }

                print("Optimization Finished!");
                // SaveModel(sess);

                // Test model
                var correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1));
                // Calculate accuracy
                var accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32));
                float acc = accuracy.eval(sess, (x, mnist.Test.Data), (y, mnist.Test.Labels));
                print($"Accuracy: {acc:F4}");

                return acc > 0.9;
            }
        }

        public void PrepareData()
        {
            mnist = MnistModelLoader.LoadAsync(".resources/mnist", oneHot: true, trainSize: train_size, validationSize: validation_size, testSize: test_size, showProgressInConsole: true).Result;
        }

        public void SaveModel(Session sess)
        {
            var saver = tf.train.Saver();
            var save_path = saver.save(sess, ".resources/logistic_regression/model.ckpt");
            tf.train.write_graph(sess.graph, ".resources/logistic_regression", "model.pbtxt", as_text: true);

            FreezeGraph.freeze_graph(input_graph: ".resources/logistic_regression/model.pbtxt",
                              input_saver: "",
                              input_binary: false,
                              input_checkpoint: ".resources/logistic_regression/model.ckpt",
                              output_node_names: "Softmax",
                              restore_op_name: "save/restore_all",
                              filename_tensor_name: "save/Const:0",
                              output_graph: ".resources/logistic_regression/model.pb",
                              clear_devices: true,
                              initializer_nodes: "");
        }

        public void Predict(Session sess)
        {
            var graph = new Graph().as_default();
            graph.Import(Path.Join(".resources/logistic_regression", "model.pb"));

            // restoring the model
            // var saver = tf.train.import_meta_graph("logistic_regression/tensorflowModel.ckpt.meta");
            // saver.restore(sess, tf.train.latest_checkpoint('logistic_regression'));
            var pred = graph.OperationByName("Softmax");
            var output = pred.outputs[0];
            var x = graph.OperationByName("Placeholder");
            var input = x.outputs[0];

            // predict
            var (batch_xs, batch_ys) = mnist.Train.GetNextBatch(10);
            var results = sess.run(output, new FeedItem(input, batch_xs[np.arange(1)]));

            if (results[0].argmax() == (batch_ys[0] as NDArray).argmax())
                print("predicted OK!");
            else
                throw new ValueError("predict error, should be 90% accuracy");
        }

        public Graph ImportGraph()
        {
            throw new NotImplementedException();
        }

        public Graph BuildGraph()
        {
            throw new NotImplementedException();
        }

        public void Train(Session sess)
        {
            throw new NotImplementedException();
        }

        public void Test(Session sess)
        {
            throw new NotImplementedException();
        }
    }
}
