using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
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

            Tensor init_loss = loss(model(inputs), outputs);
            // print($"Current loss: {init_loss.numpy()}");

            // Define a training loop
            Func<Tensor, Tensor, float, Tensor> train = (inputs, outputs, learning_rate)
                =>
                {
                    using var t = tf.GradientTape();
                    var current_loss = loss(outputs, model(inputs));
                    var (dW, db) = t.gradient(current_loss, (W, b));
                    W.assign_sub(learning_rate * dW);
                    b.assign_sub(learning_rate * db);
                    return current_loss;
                };

            var epochs = range(10);
            foreach (var epoch in epochs)
            {
                var current_loss = train(inputs, outputs, 0.1f);
                print($"Epoch {epoch}: W={(float)W.numpy()} b={(float)b.numpy()}, loss={(float)current_loss.numpy()}");

                if (epoch > 0) // skip first epoch
                    Assert.IsTrue((bool)(current_loss < init_loss));
            }
        }
    }
}
