﻿using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow;
using static Tensorflow.Python;

namespace TensorFlowNET.UnitTest.control_flow_ops_test
{
    /// <summary>
    /// excerpt of tensorflow/python/framework/ops/control_flow_ops_test.py
    /// </summary>
    [TestClass]
    public class CondTestCases : PythonTest
    {
        [Ignore("need tesnroflow expose AddControlInput API")]
        [TestMethod]
        public void testCondTrue_ConstOnly()
        {
            var graph = tf.Graph().as_default();

            with(tf.Session(graph), sess =>
            {
                var x = tf.constant(2, name: "x");
                var y = tf.constant(5, name: "y");

                var z = control_flow_ops.cond(tf.less(x, y),
                    () => tf.constant(22, name: "t22"),
                    () => tf.constant(55, name: "f55"));

                int result = z.eval(sess);
                assertEquals(result, 22);
            });
        }

        [Ignore("need tesnroflow expose AddControlInput API")]
        [TestMethod]
        public void testCondFalse_ConstOnly()
        {
            var graph = tf.Graph().as_default();

            with(tf.Session(graph), sess =>
            {
                var x = tf.constant(2, name: "x");
                var y = tf.constant(1, name: "y");

                var z = control_flow_ops.cond(tf.less(x, y),
                    () => tf.constant(22, name: "t22"),
                    () => tf.constant(11, name: "f11"));

                int result = z.eval(sess);
                assertEquals(result, 11);
            });
        }

        [Ignore("need tesnroflow expose AddControlInput API")]
        [TestMethod]
        public void testCondTrue()
        {
            tf.Graph().as_default();

            var x = tf.constant(2, name: "x");
            var y = tf.constant(5, name: "y");

            var z = control_flow_ops.cond(tf.less(x, y),
                () => tf.multiply(x, 17),
                () => tf.add(y, 23));

            var result = evaluate<int>(z);
            assertEquals(result, 34);
        }

        [Ignore("need tesnroflow expose AddControlInput API")]
        [TestMethod]
        public void testCondFalse()
        {
            tf.Graph().as_default();

            var x = tf.constant(2);
            var y = tf.constant(1);

            var z = control_flow_ops.cond(tf.less(x, y),
                () => tf.multiply(x, 17),
                () => tf.add(y, 23));

            var result = evaluate<int>(z);
            assertEquals(result, 24);
        }

        // NOTE: all other python test cases of this class are either not needed due to strong typing or test a deprecated api

    }
}
