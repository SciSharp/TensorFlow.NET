using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// A linear regression learning algorithm example using TensorFlow library.
    /// https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/linear_regression.py
    /// </summary>
    public class LinearRegression : IExample
    {
        private NumPyRandom rng = np.random;

        public void Run()
        {
            // Parameters
            double learning_rate = 0.01;
            int training_epochs = 1000;
            int display_step = 50;

            // Training Data
            var train_X = np.array(3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                         7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1);
            var train_Y = np.array(1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                         2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3);
            var n_samples = train_X.shape[0];
            
            // tf Graph Input
            var X = tf.placeholder(tf.float64);
            var Y = tf.placeholder(tf.float64);

            // Set model weights 
            var W = tf.Variable(rng.randn<double>(), name: "weight");
            var b = tf.Variable(rng.randn<double>(), name: "bias");

            var mul = tf.multiply(X, W);
            var pred = tf.add(mul, b);

            // Mean squared error
            var sub = pred - Y;
            var pow = tf.pow(sub, 2);

            var reduce = tf.reduce_sum(pow);
            var cost = reduce / (2d * n_samples);

            // radient descent
            // Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
            var grad = tf.train.GradientDescentOptimizer(learning_rate);
            var optimizer = grad.minimize(cost);

            // Initialize the variables (i.e. assign their default value)
            var init = tf.global_variables_initializer();

            // Start training
            Python.with<Session>(tf.Session(), sess => 
            {
                // Run the initializer
                sess.run(init);

                // Fit all training data
                for (int epoch = 0; epoch < training_epochs; epoch++)
                {
                    foreach (var (x, y) in Python.zip<double>(train_X, train_Y))
                    {
                        sess.run(optimizer,
                            new FeedItem(X, x),
                            new FeedItem(Y, y));
                    }

                    // Display logs per epoch step
                    if ((epoch + 1) % display_step == 0)
                    {
                        var c = sess.run(cost, 
                            new FeedItem(X, train_X),
                            new FeedItem(Y, train_Y));
                        var rW = sess.run(W);
                        Console.WriteLine($"Epoch: {epoch + 1} cost={c} " +
                                    $"W={rW} b={sess.run(b)}");
                    }
                }

                Console.WriteLine("Optimization Finished!");
            });
        }
    }
}
