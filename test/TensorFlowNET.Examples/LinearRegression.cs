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
    public class LinearRegression : Python, IExample
    {
        private NumPyRandom rng = np.random;

        public void Run()
        {
            var graph = tf.Graph().as_default();

            // Parameters
            float learning_rate = 0.01f;
            int training_epochs = 1000;
            int display_step = 10;

            // Training Data
            var train_X = np.array(3.3f, 4.4f, 5.5f, 6.71f, 6.93f, 4.168f, 9.779f, 6.182f, 7.59f, 2.167f,
                         7.042f, 10.791f, 5.313f, 7.997f, 5.654f, 9.27f, 3.1f);
            var train_Y = np.array(1.7f, 2.76f, 2.09f, 3.19f, 1.694f, 1.573f, 3.366f, 2.596f, 2.53f, 1.221f,
                         2.827f, 3.465f, 1.65f, 2.904f, 2.42f, 2.94f, 1.3f);
            var n_samples = train_X.shape[0];

            // tf Graph Input
            /*var X = tf.placeholder(tf.float32);
            var Y = tf.placeholder(tf.float32);

            // Set model weights 
            //var rnd1 = rng.randn<float>();
            //var rnd2 = rng.randn<float>();
            var W = tf.Variable(-0.06f, name: "weight");
            var b = tf.Variable(-0.73f, name: "bias");

            var mul = tf.multiply(X, W);
            var pred = tf.add(mul, b);

            // Mean squared error
            var sub = pred - Y;
            var pow = tf.pow(sub, 2.0f);

            var reduce = tf.reduce_sum(pow);
            var cost = reduce / (2.0f * n_samples);

            // import graph

            // radient descent
            // Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
            var grad = tf.train.GradientDescentOptimizer(learning_rate);
            var optimizer = grad.minimize(cost);*/

            var new_saver = tf.train.import_meta_graph("linear_regression.meta");

            var X = graph.OperationByName("Placeholder");
            var Y = graph.OperationByName("Placeholder_1");
            var W = graph.OperationByName("weight");
            var optimizer = graph.OperationByName("GradientDescent");

            // Initialize the variables (i.e. assign their default value)
            var init = tf.global_variables_initializer();

            // Start training
            Python.with<Session>(tf.Session(graph), sess => 
            {
                // Run the initializer
                sess.run(init);

                // Fit all training data
                for (int epoch = 0; epoch < training_epochs; epoch++)
                {
                    foreach (var (x, y) in zip<float>(train_X, train_Y))
                    {
                        var w = sess.run(W);
                        sess.run(optimizer,
                            new FeedItem(X, x),
                            new FeedItem(Y, y));
                        w = sess.run(W);
                    }

                    // Display logs per epoch step
                    /*if ((epoch + 1) % display_step == 0)
                    {
                        var c = sess.run(cost, 
                            new FeedItem(X, train_X),
                            new FeedItem(Y, train_Y));
                        var rW = sess.run(W);
                        Console.WriteLine($"Epoch: {epoch + 1} cost={c} " +
                                    $"W={rW} b={sess.run(b)}");
                    }*/
                }

                Console.WriteLine("Optimization Finished!");
            });
        }
    }
}
