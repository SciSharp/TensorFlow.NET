using Newtonsoft.Json;
using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow;
using TensorFlowNET.Examples.Utility;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// A logistic regression learning algorithm example using TensorFlow library.
    /// This example is using the MNIST database of handwritten digits
    /// https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/logistic_regression.py
    /// </summary>
    public class LogisticRegression : Python, IExample
    {
        private float learning_rate = 0.01f;
        private int training_epochs = 5;
        private int batch_size = 100;
        private int display_step = 1;

        Datasets mnist;

        public void Run()
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

            with(tf.Session(), sess =>
            {
                
                // Run the initializer
                sess.run(init);

                // Training cycle
                foreach (var epoch in range(training_epochs))
                {
                    var avg_cost = 0.0f;
                    var total_batch = mnist.train.num_examples / batch_size;
                    // Loop over all batches
                    foreach (var i in range(total_batch))
                    {
                        var (batch_xs, batch_ys) = mnist.train.next_batch(batch_size);
                        // Run optimization op (backprop) and cost op (to get loss value)
                        var result = sess.run(new object[] { optimizer, cost },
                            new FeedItem(x, batch_xs),
                            new FeedItem(y, batch_ys));

                        var c = (float)result[1];
                        // Compute average loss
                        avg_cost += c / total_batch;
                    }

                    // Display logs per epoch step
                    if ((epoch + 1) % display_step == 0)
                        print($"Epoch: {(epoch + 1).ToString("D4")} cost= {avg_cost.ToString("G9")}");
                }

                print("Optimization Finished!");

                // Test model
                var correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1));
                // Calculate accuracy
                var accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32));
                float acc = accuracy.eval(new FeedItem(x, mnist.test.images), new FeedItem(y, mnist.test.labels));
                print($"Accuracy: {acc.ToString("F4")}");
            });
        }

        public void PrepareData()
        {
            mnist = MnistDataSet.read_data_sets("logistic_regression", one_hot: true);
        }
    }
}
