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

            var part1 = tf.multiply(X, W);
            var pred = tf.add(part1, b);

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
                for (int i = 0; i < training_epochs; i++)
                {
                    for(int index = 0; index < train_X.size; index++)
                    {
                        (double x, double y) = Python.zip<double>(train_X, train_Y, index);
                        var feed_dict = new Dictionary<Tensor, NDArray>();

                        //sess.run(optimizer, feed_dict);
                    }
                }
            });
        }
    }
}
