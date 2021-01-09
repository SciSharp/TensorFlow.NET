using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using Tensorflow;
using Tensorflow.Graphs;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.ManagedAPI
{
    [TestClass]
    public class FunctionApiTest : EagerModeTestBase
    {
        Tensor Min(Tensor a, Tensor b)
        {
            return tf.cond(a < b, () => a, () => b);
        }

        [TestMethod]
        public void MulInAutoGraph()
        {
            var a = tf.constant(1);
            var b = tf.constant(2);
            // For first time running, tf.net will record the operations in graph mode.
            // And register to tensorflow op library. 
            var output = Mul(a, b);
            Assert.AreEqual(2, (int)output);

            var c = tf.constant(3);
            // for the following invoke, Mul will be intercepted and run it in eager mode.
            output = Mul(b, c);
            Assert.AreEqual(6, (int)output);
        }

        /// <summary>
        /// Method with AutoGraph attribute will be converted to FuncGraph
        /// when it's invoked for the first time.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        [AutoGraph]
        Tensor Mul(Tensor a, Tensor b)
        {
            return a * b;
        }

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
