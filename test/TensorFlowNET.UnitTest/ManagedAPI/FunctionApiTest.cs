using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow;
using Tensorflow.Graphs;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.ManagedAPI
{
    [TestClass]
    public class FunctionApiTest : TFNetApiTest
    {
        [TestMethod]
        public void TwoInputs_OneOutput()
        {
            var func = tf.autograph.to_graph(Add);
            var a = tf.constant(1);
            var b = tf.constant(2);
            var output = func(a, b);
            Assert.AreEqual(3, (int)output);
        }

        Tensor Add(Tensor a, Tensor b)
        {
            return a + b;
        }

        [TestMethod]
        public void TwoInputs_OneOutput_Condition()
        {
            var func = tf.autograph.to_graph(Condition);
            var a = tf.constant(3);
            var b = tf.constant(2);
            var output = func(a, b);
            Assert.AreEqual(2, (int)output);
        }

        Tensor Condition(Tensor a, Tensor b)
        {
            return tf.cond(a < b, a, b);
        }

        [TestMethod]
        public void TwoInputs_OneOutput_Lambda()
        {
            var func = tf.autograph.to_graph((x, y) => x * y);
            var output = func(tf.constant(3), tf.constant(2));
            Assert.AreEqual(6, (int)output);
        }

        [TestMethod]
        public void TwoInputs_OneOutput_WhileLoop()
        {
            var func = tf.autograph.to_graph((x, y) => x * y);
            var output = func(tf.constant(3), tf.constant(2));
            Assert.AreEqual(6, (int)output);
        }

        Tensor WhileLoop()
        {
            var i = tf.constant(0);
            Func<Tensor, Tensor> c = i => tf.less(i, 10);
            Func<Tensor, Tensor> b = i => tf.add(i, 1);
            //var r = tf.(c, b, [i])
            throw new NotImplementedException("");
        }
    }
}
