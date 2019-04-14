using Microsoft.VisualStudio.TestTools.UnitTesting;
using Newtonsoft.Json;
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
            // tf.train.import_meta_graph("cond_test.meta");
            var json = JsonConvert.SerializeObject(graph._nodes_by_name, Formatting.Indented);

            with(tf.Session(graph), sess =>
            {
                var x = tf.constant(2, name: "x"); // graph.get_operation_by_name("Const").output; 
                var y = tf.constant(5, name: "y"); // graph.get_operation_by_name("Const_1").output;
                var pred = tf.less(x, y); // graph.get_operation_by_name("Less").output;

                Func<ITensorOrOperation> if_true = delegate
                {
                    return tf.constant(2, name: "t2");
                };

                Func<ITensorOrOperation> if_false = delegate
                {
                    return tf.constant(5, name: "f5");
                };

                var z = control_flow_ops.cond(pred, if_true, if_false); // graph.get_operation_by_name("cond/Merge").output

                json = JsonConvert.SerializeObject(graph._nodes_by_name, Formatting.Indented);
                int result = z.eval(sess);
                assertEquals(result, 2);
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

            var graph = tf.Graph().as_default();
            //tf.train.import_meta_graph("cond_test.meta");
            //var json = JsonConvert.SerializeObject(graph._nodes_by_name, Formatting.Indented);

            with(tf.Session(), sess =>
            {
                var x = tf.constant(2, name: "x");
                var y = tf.constant(1, name: "y");
                var pred = tf.less(x, y);

                Func<ITensorOrOperation> if_true = delegate
                {
                    return tf.constant(2, name: "t2");
                };

                Func<ITensorOrOperation> if_false = delegate
                {
                    return tf.constant(1, name: "f1");
                };

                var z = control_flow_ops.cond(pred, if_true, if_false);

                var json1 = JsonConvert.SerializeObject(graph._nodes_by_name, Formatting.Indented);
                int result = z.eval(sess);
                assertEquals(result, 1);
            });
        }

        [Ignore("Todo")]
        [TestMethod]
        public void testCondTrueLegacy()
        {
            //  def testCondTrueLegacy(self):
            //    x = constant_op.constant(2)
            //    y = constant_op.constant(5)
            //    z = control_flow_ops.cond(
            //        math_ops.less(x, y),
            //        fn1=lambda: math_ops.multiply(x, 17),
            //        fn2=lambda: math_ops.add(y, 23))
            //    self.assertEquals(self.evaluate(z), 34)
        }

        [Ignore("Todo")]
        [TestMethod]
        public void testCondFalseLegacy()
        {
            //  def testCondFalseLegacy(self):
            //    x = constant_op.constant(2)
            //    y = constant_op.constant(1)
            //    z = control_flow_ops.cond(
            //        math_ops.less(x, y),
            //        fn1=lambda: math_ops.multiply(x, 17),
            //        fn2=lambda: math_ops.add(y, 23))
            //    self.assertEquals(self.evaluate(z), 24)
        }

        [Ignore("Todo")]
        [TestMethod]
        public void testCondMissingArg1()
        {
            //  def testCondMissingArg1(self):
            //    x = constant_op.constant(1)
            //    with self.assertRaises(TypeError):
            //      control_flow_ops.cond(True, false_fn=lambda: x)

        }

        [Ignore("Todo")]
        [TestMethod]
        public void testCondMissingArg2()
        {
            //  def testCondMissingArg2(self):
            //    x = constant_op.constant(1)
            //    with self.assertRaises(TypeError):
            //      control_flow_ops.cond(True, lambda: x)
        }

        [Ignore("Todo")]
        [TestMethod]
        public void testCondDuplicateArg1()
        {
            //  def testCondDuplicateArg1(self):
            //    x = constant_op.constant(1)
            //    with self.assertRaises(TypeError):
            //      control_flow_ops.cond(True, lambda: x, lambda: x, fn1=lambda: x)
        }

        [Ignore("Todo")]
        [TestMethod]
        public void testCondDuplicateArg2()
        {
            //  def testCondDuplicateArg2(self):
            //    x = constant_op.constant(1)
            //    with self.assertRaises(TypeError):
            //      control_flow_ops.cond(True, lambda: x, lambda: x, fn2=lambda: x)
        }

    }
}
