using NumSharp.Core;
using System;
using System.Collections.Generic;
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
           var mnist = MnistDataSet.read_data_sets("logistic_regression", one_hot: true);

            // tf Graph Input
            var x = tf.placeholder(tf.float32, new TensorShape(-1, 784)); // mnist data image of shape 28*28=784
            var y = tf.placeholder(tf.float32, new TensorShape(-1, 10)); // 0-9 digits recognition => 10 classes

            // Set model weights
            var W = tf.Variable(tf.zeros(new Shape(784, 10)));
            var b = tf.Variable(tf.zeros(new Shape(10)));

            // Construct model
            var pred = tf.nn.softmax(tf.matmul(x, W) + b); // Softmax

            // Minimize error using cross entropy
            var sum = -tf.reduce_sum(y * tf.log(pred), reduction_indices: 1);
            var cost = tf.reduce_mean(sum);

            // Gradient Descent
            var optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost);

            // Initialize the variables (i.e. assign their default value)
            var init = tf.global_variables_initializer();

            with(tf.Session(), sess =>
            {
                // Run the initializer
                sess.run(init);
            });
        }
    }
}
