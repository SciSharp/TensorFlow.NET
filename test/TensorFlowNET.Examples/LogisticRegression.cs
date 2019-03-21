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
        private int training_epochs = 25;
        private int batch_size = 100;
        private int display_step = 1;

        public void Run()
        {
            PrepareData();
        }

        private void PrepareData()
        {
            // tf Graph Input
            var x = tf.placeholder(tf.float32, new TensorShape(-1, 784)); // mnist data image of shape 28*28=784
            var y = tf.placeholder(tf.float32, new TensorShape(-1, 10)); // 0-9 digits recognition => 10 classes

            // Set model weights
            var W = tf.Variable(tf.zeros(new Shape(784, 10)));
            var b = tf.Variable(tf.zeros(new Shape(10)));

            // Construct model
            var pred = tf.nn.softmax(tf.matmul(x, W) + b); // Softmax

            // Minimize error using cross entropy
            var log = tf.log(pred);
            var mul = y * log;
            var sum = tf.reduce_sum(mul, reduction_indices: 1);
            var neg = -sum;
            var cost = tf.reduce_mean(neg);

            // Gradient Descent
            var optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost);

            // Initialize the variables (i.e. assign their default value)
            var init = tf.global_variables_initializer();

            with(tf.Session(), sess =>
            {
                var mnist = MnistDataSet.read_data_sets("logistic_regression", one_hot: true);
                // Run the initializer
                sess.run(init);

                // Training cycle
                foreach(var epoch in range(training_epochs))
                {
                    var avg_cost = 0.0f;
                    var total_batch = (int)(mnist.train.num_examples / batch_size);
                    // Loop over all batches
                    foreach (var i in range(total_batch))
                    {
                        var (batch_xs, batch_ys) = mnist.train.next_batch(batch_size);
                        // Run optimization op (backprop) and cost op (to get loss value)
                        /*sess.run(optimizer,
                            new FeedItem(x, batch_xs),
                            new FeedItem(y, batch_ys));*/
                    }
                }
            });
        }
    }
}
