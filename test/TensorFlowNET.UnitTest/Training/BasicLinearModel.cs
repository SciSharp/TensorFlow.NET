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
        int NUM_EXAMPLES = 1000;

        [TestMethod]
        public void FitLinear()
        {
            // Initialize the weights to `5.0` and the bias to `0.0`
            // In practice, these should be initialized to random values (for example, with `tf.random.normal`)
            var W = tf.Variable(5.0f);
            var b = tf.Variable(0.0);

            // define linear model
            Func<NDArray, Tensor> model = (x) => W * x + b;

            // var inputs = tf.random.normal(shape =[NUM_EXAMPLES]);
            // noise = tf.random.normal(shape =[NUM_EXAMPLES])
            // outputs = inputs * TRUE_W + TRUE_b + noise
        }
    }
}
