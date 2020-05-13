using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using System.Linq;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.Basics
{
    [TestClass]
    public class VariableTest
    {
        [Ignore]
        [TestMethod]
        public void NewVariable()
        {
            var x = tf.Variable(10, name: "new_variable_x");
            Assert.AreEqual("new_variable_x:0", x.Name);
            Assert.AreEqual(0, x.shape.ndim);
            Assert.AreEqual(10, (int)x.numpy());
        }

        [TestMethod]
        public void StringVar()
        {
            var mammal1 = tf.Variable("Elephant", name: "var1", dtype: tf.@string);
            var mammal2 = tf.Variable("Tiger");
        }

        [TestMethod]
        public void VarSum()
        {
            var x = tf.constant(3, name: "x");
            var y = tf.Variable(x + 1, name: "y");
            Assert.AreEqual(4, (int)y.numpy());
        }

        [TestMethod]
        public void Assign1()
        {
            var variable = tf.Variable(31, name: "tree");
            var unread = variable.assign(12);
            Assert.AreEqual(12, (int)unread.numpy());
        }

        [TestMethod]
        public void Assign2()
        {
            var v1 = tf.Variable(10.0f, name: "v1");
            var v2 = v1.assign(v1 + 1.0f);
            Assert.AreEqual(v1.numpy(), v2.numpy());
            Assert.AreEqual(11f, (float)v1.numpy());
        }

        [TestMethod]
        public void Accumulation()
        {
            var x = tf.Variable(10, name: "x");
            /*for (int i = 0; i < 5; i++)
                x = x + 1;

            Assert.AreEqual(15, (int)x.numpy());*/
        }

        [TestMethod]
        public void ShouldReturnNegative()
        {
            var x = tf.constant(new[,] { { 1, 2 } });
            var neg_x = tf.negative(x);
            Assert.IsTrue(Enumerable.SequenceEqual(new[] { 1, 2 }, neg_x.shape));
            Assert.IsTrue(Enumerable.SequenceEqual(new[] { -1, -2 }, neg_x.numpy().ToArray<int>()));
        }
    }
}
