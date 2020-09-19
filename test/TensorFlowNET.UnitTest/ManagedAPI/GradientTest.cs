using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using System.Linq;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.ManagedAPI
{
    [TestClass]
    public class GradientTest
    {
        [TestMethod]
        public void GradientFloatTest()
        {
            var x = tf.Variable(3.0, dtype: tf.float32);
            using var tape = tf.GradientTape();
            var y = tf.square(x);
            var y_grad = tape.gradient(y, x);
            Assert.AreEqual(9.0f, (float)y);
        }

        [TestMethod]
        public void GradientDefaultTest()
        {
            var x = tf.Variable(3.0);
            using var tape = tf.GradientTape();
            var y = tf.square(x);
            var y_grad = tape.gradient(y, x);
            Assert.AreEqual(9.0, (double)y);
        }

        [TestMethod]
        public void GradientDoubleTest()
        {
            var x = tf.Variable(3.0, dtype: tf.float64);
            using var tape = tf.GradientTape();
            var y = tf.square(x);
            var y_grad = tape.gradient(y, x);
            Assert.AreEqual(9.0, (double)y);
        }
    }
}
