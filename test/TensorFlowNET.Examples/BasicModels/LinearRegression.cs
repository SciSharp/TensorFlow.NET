using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;
using static Tensorflow.Python;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// A linear regression learning algorithm example using TensorFlow library.
    /// https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/linear_regression.py
    /// </summary>
    public class LinearRegression : IExample
    {
        public bool Enabled { get; set; } = true;
        public string Name => "Linear Regression";
        public bool IsImportingGraph { get; set; } = false;

        public int training_epochs = 1000;

        // Parameters
        float learning_rate = 0.01f;
        int display_step = 50;

        NumPyRandom rng = np.random;
        NDArray train_X, train_Y;
        int n_samples;

        public bool Run()
        {
            // Training Data
            PrepareData();

            // tf Graph Input
            var X = tf.placeholder(tf.float32);
            var Y = tf.placeholder(tf.float32);

            // Set model weights 
            // We can set a fixed init value in order to debug
            // var rnd1 = rng.randn<float>();
            // var rnd2 = rng.randn<float>();
            var W = tf.Variable(-0.06f, name: "weight");
            var b = tf.Variable(-0.73f, name: "bias");

            // Construct a linear model
            var pred = tf.add(tf.multiply(X, W), b);

            // Mean squared error
            var cost = tf.reduce_sum(tf.pow(pred - Y, 2.0f)) / (2.0f * n_samples);

            // Gradient descent
            // Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
            var optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost);

            // Initialize the variables (i.e. assign their default value)
            var init = tf.global_variables_initializer();

            // Start training
            return with(tf.Session(), sess => 
            {
                // Run the initializer
                sess.run(init);

                // Fit all training data
                for (int epoch = 0; epoch < training_epochs; epoch++)
                {
                    foreach (var (x, y) in zip<float>(train_X, train_Y))
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
                        Console.WriteLine($"Epoch: {epoch + 1} cost={c} " + $"W={sess.run(W)} b={sess.run(b)}");
                    }
                }

                Console.WriteLine("Optimization Finished!");
                var training_cost = sess.run(cost,
                    new FeedItem(X, train_X),
                    new FeedItem(Y, train_Y));
                Console.WriteLine($"Training cost={training_cost} W={sess.run(W)} b={sess.run(b)}");

                // Testing example
                var test_X = np.array(6.83f, 4.668f, 8.9f, 7.91f, 5.7f, 8.7f, 3.1f, 2.1f);
                var test_Y = np.array(1.84f, 2.273f, 3.2f, 2.831f, 2.92f, 3.24f, 1.35f, 1.03f);
                Console.WriteLine("Testing... (Mean square loss Comparison)");
                var testing_cost = sess.run(tf.reduce_sum(tf.pow(pred - Y, 2.0f)) / (2.0f * test_X.shape[0]),
                    new FeedItem(X, test_X), 
                    new FeedItem(Y, test_Y));
                Console.WriteLine($"Testing cost={testing_cost}");
                var diff = Math.Abs((float)training_cost - (float)testing_cost);
                Console.WriteLine($"Absolute mean square loss difference: {diff}");

                return diff < 0.01;
            });
        }

        public void PrepareData()
        {
            train_X = np.array(3.3f, 4.4f, 5.5f, 6.71f, 6.93f, 4.168f, 9.779f, 6.182f, 7.59f, 2.167f,
             7.042f, 10.791f, 5.313f, 7.997f, 5.654f, 9.27f, 3.1f);
            train_Y = np.array(1.7f, 2.76f, 2.09f, 3.19f, 1.694f, 1.573f, 3.366f, 2.596f, 2.53f, 1.221f,
                         2.827f, 3.465f, 1.65f, 2.904f, 2.42f, 2.94f, 1.3f);
            n_samples = train_X.shape[0];
        }

        public Graph ImportGraph()
        {
            throw new NotImplementedException();
        }

        public Graph BuildGraph()
        {
            throw new NotImplementedException();
        }

        public bool Train()
        {
            throw new NotImplementedException();
        }

        public bool Predict()
        {
            throw new NotImplementedException();
        }
    }
}
