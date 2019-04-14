using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using Tensorflow;

namespace TensorFlowNET.UnitTest.control_flow_ops_test
{
    /// <summary>
    /// excerpt of tensorflow/python/framework/ops/control_flow_ops_test.py
    /// </summary>
    [TestClass]
    public class CondTestCases : PythonTest
    {
        [TestMethod]
        public void testCondTrue()
        {
            var graph = tf.Graph().as_default();

            with(tf.Session(graph), sess =>
            {
                var x = tf.constant(2);
                var y = tf.constant(5);
                var pred = tf.less(x, y);

                Func<ITensorOrOperation> if_true = delegate
                {
                    return tf.multiply(x, 17);
                };

                Func<ITensorOrOperation> if_false = delegate
                {
                    return tf.add(y, 23);
                };

                var z = control_flow_ops.cond(pred, if_true, if_false);
                int result = z.eval(sess);
                assertEquals(result, 34);
            });
        }

        [TestMethod]
        public void testCondFalse()
        {
            /* python
             * import tensorflow as tf
               from tensorflow.python.framework import ops

                def if_true():
                    return tf.math.multiply(x, 17)
                def if_false():
                    return tf.math.add(y, 23)

                with tf.Session() as sess:
                    x = tf.constant(2)
                    y = tf.constant(1)
                    pred = tf.math.less(x,y)
                    z = tf.cond(pred, if_true, if_false)
                    result = z.eval()

                print(result == 24) */

            with(tf.Session(), sess =>
            {
                var x = tf.constant(2);
                var y = tf.constant(1);
                var pred = tf.less(x, y);

                Func<ITensorOrOperation> if_true = delegate
                {
                    return tf.multiply(x, 17);
                };

                Func<ITensorOrOperation> if_false = delegate
                {
                    return tf.add(y, 23);
                };

                var z = control_flow_ops.cond(pred, if_true, if_false);
                int result = z.eval(sess);
                assertEquals(result, 24);
            });
        }

        // NOTE: all other test python test cases of this class are either not needed due to strong typing or dest a deprecated api

    }
}
