using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using System.Linq;
using Tensorflow;
using static Tensorflow.Binding;

namespace Tensorflow.UnitTest.TF_API
{
    [TestClass]
    public class GradientTest
    {
        [TestMethod]
        public void GradientFloatTest()
        {
            var x = tf.Variable(3.0, dtype: TF_DataType.TF_FLOAT);
            using var tape = tf.GradientTape();
            var y = tf.square(x);
            var y_grad = tape.gradient(y, x);
            Assert.AreEqual(9.0f, (float)y);
        }

        [TestMethod]
        public void GradientDefaultTest()
        {//error 1#: Variable default type 
            var x = tf.Variable(3.0);
            using var tape = tf.GradientTape();
            var y = tf.square(x);
            var y_grad = tape.gradient(y, x);
            Assert.AreEqual(9.0, (double)y);
        }
        [TestMethod]
        public void GradientDoubleTest()
        {//error 2#: Variable double type
            var x = tf.Variable(3.0, dtype: TF_DataType.TF_DOUBLE);
            using var tape = tf.GradientTape();
            var y = tf.square(x);
            var y_grad = tape.gradient(y, x);
            Assert.AreEqual(9.0, (double)y);
        }





    }
}
