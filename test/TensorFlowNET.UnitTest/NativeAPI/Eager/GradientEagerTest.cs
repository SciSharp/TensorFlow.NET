using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.Gradient
{
    [TestClass]
    public class GradientEagerTest : PythonTest
    {
        [Ignore]
        [TestMethod]
        public void ConstantSq()
        {
            // Calcute the gradient of w * w 
            // by Automatic Differentiation in Eager mode
            // in tensorflow.net 2.x that is in development intensively
            var w = tf.constant(1.5f);
            using var tape = tf.GradientTape();
            tape.watch(w);
            var loss = w * w;
            var grad = tape.gradient(loss, w);
            print(grad);
        }
    }
}
