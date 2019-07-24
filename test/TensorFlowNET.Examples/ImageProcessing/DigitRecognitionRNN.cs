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
using Tensorflow;
using TensorFlowNET.Examples.Utility;
using static Tensorflow.Python;

namespace TensorFlowNET.Examples.ImageProcess
{
    /// <summary>
    /// Recurrent Neural Network for handwritten digits MNIST.
    /// https://medium.com/machine-learning-algorithms/mnist-using-recurrent-neural-network-2d070a5915a2
    /// </summary>
    public class DigitRecognitionRNN : IExample
    {
        public bool Enabled { get; set; } = false;
        public bool IsImportingGraph { get; set; } = false;

        public string Name => "MNIST RNN";

        string logs_path = "logs";

        // Hyper-parameters
        int n_neurons = 128;
        float learning_rate = 0.001f;
        int batch_size = 128;
        int epochs = 10;

        int n_steps = 28;
        int n_inputs = 28;
        int n_outputs = 10;

        Datasets<DataSetMnist> mnist;

        Tensor x, y;
        Tensor loss, accuracy, cls_prediction;
        Operation optimizer;

        int display_freq = 100;
        float accuracy_test = 0f;
        float loss_test = 1f;

        NDArray x_train, y_train;
        NDArray x_valid, y_valid;
        NDArray x_test, y_test;

        public bool Run()
        {
            PrepareData();
            BuildGraph();

            with(tf.Session(), sess =>
            {
                Train(sess);
                Test(sess);
            });

            return loss_test < 0.09 && accuracy_test > 0.95;
        }

        public Graph BuildGraph()
        {
            var graph = new Graph().as_default();

            var X = tf.placeholder(tf.float32, new[] { -1, n_steps, n_inputs });
            var y = tf.placeholder(tf.int32, new[] { -1 });
            var cell = tf.nn.rnn_cell.BasicRNNCell(num_units: n_neurons);
            var (output, state) = tf.nn.dynamic_rnn(cell, X, dtype: tf.float32);

            return graph;
        }

        public void Train(Session sess)
        {
            // Number of training iterations in each epoch
            var num_tr_iter = y_train.len / batch_size;

            var init = tf.global_variables_initializer();
            sess.run(init);

            float loss_val = 100.0f; 
            float accuracy_val = 0f;

            foreach (var epoch in range(epochs))
            {
                print($"Training epoch: {epoch + 1}");
                // Randomly shuffle the training data at the beginning of each epoch 
                (x_train, y_train) = mnist.Randomize(x_train, y_train);

                foreach (var iteration in range(num_tr_iter))
                {
                    var start = iteration * batch_size;
                    var end = (iteration + 1) * batch_size;
                    var (x_batch, y_batch) = mnist.GetNextBatch(x_train, y_train, start, end);

                    // Run optimization op (backprop)
                    sess.run(optimizer, new FeedItem(x, x_batch), new FeedItem(y, y_batch));

                    if (iteration % display_freq == 0)
                    {
                        // Calculate and display the batch loss and accuracy
                        var result = sess.run(new[] { loss, accuracy }, new FeedItem(x, x_batch), new FeedItem(y, y_batch));
                        loss_val = result[0];
                        accuracy_val = result[1];
                        print($"iter {iteration.ToString("000")}: Loss={loss_val.ToString("0.0000")}, Training Accuracy={accuracy_val.ToString("P")}");
                    }
                }

                // Run validation after every epoch
                var results1 = sess.run(new[] { loss, accuracy }, new FeedItem(x, x_valid), new FeedItem(y, y_valid));
                loss_val = results1[0];
                accuracy_val = results1[1];
                print("---------------------------------------------------------");
                print($"Epoch: {epoch + 1}, validation loss: {loss_val.ToString("0.0000")}, validation accuracy: {accuracy_val.ToString("P")}");
                print("---------------------------------------------------------");
            }
        }

        public void Test(Session sess)
        {
            var result = sess.run(new[] { loss, accuracy }, new FeedItem(x, x_test), new FeedItem(y, y_test));
            loss_test = result[0];
            accuracy_test = result[1];
            print("---------------------------------------------------------");
            print($"Test loss: {loss_test.ToString("0.0000")}, test accuracy: {accuracy_test.ToString("P")}");
            print("---------------------------------------------------------");
        }

        public void PrepareData()
        {
            mnist = MNIST.read_data_sets("mnist", one_hot: true);
            (x_train, y_train) = (mnist.train.data, mnist.train.labels);
            (x_valid, y_valid) = (mnist.validation.data, mnist.validation.labels);
            (x_test, y_test) = (mnist.test.data, mnist.test.labels);

            print("Size of:");
            print($"- Training-set:\t\t{len(mnist.train.data)}");
            print($"- Validation-set:\t{len(mnist.validation.data)}");
            print($"- Test-set:\t\t{len(mnist.test.data)}");
        }

        public Graph ImportGraph() => throw new NotImplementedException();

        public void Predict(Session sess) => throw new NotImplementedException();
    }
}
