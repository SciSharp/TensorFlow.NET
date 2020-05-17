using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.Training
{
    [TestClass]
    public class BasicLinearModel
    {
        /// <summary>
        /// Linear Regression without tf.train.Optimizer
        /// https://www.tensorflow.org/tutorials/customization/custom_training
        /// </summary>
        [TestMethod]
        public void LinearRegression()
        {
            // Initialize the weights to `5.0` and the bias to `0.0`
            // In practice, these should be initialized to random values (for example, with `tf.random.normal`)
            var W = tf.Variable(5.0f);
            var b = tf.Variable(0.0f);

            // Define linear model
            Func<Tensor, Tensor> model = (x) => W * x + b;

            // Define the loss function
            Func<Tensor, Tensor, Tensor> loss = (target_y, predicted_y) 
                => tf.reduce_mean(tf.square(target_y - predicted_y));

            int NUM_EXAMPLES = 1000;
            float TRUE_W = 3.0f;
            float TRUE_b = 2.0f;

            var inputs = tf.random.normal(shape: NUM_EXAMPLES);
            var noise = tf.random.normal(shape: NUM_EXAMPLES);
            var outputs = inputs * TRUE_W + TRUE_b + noise;

            print($"Current loss: {loss(model(inputs), outputs).numpy()}");

            // Define a training loop
            Action<Tensor, Tensor, float> train = (inputs, outputs, learning_rate)
                =>
                {
                    using var t = tf.GradientTape();
                    var current_loss = loss(outputs, model(inputs));
                    var (dW, db) = t.gradient(current_loss, (W, b));
                    W.assign_sub(learning_rate * dW);
                    b.assign_sub(learning_rate * db);
                };

            var epochs = range(10);
            foreach(var epoch in epochs)
            {
                train(inputs, outputs, 0.1f);
                print($"Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f");
            }
        }
    }
}
