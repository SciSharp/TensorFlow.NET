using Microsoft.VisualStudio.TestTools.UnitTesting;
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
            with(tf.Graph().as_default(), g =>
            {
                var x = tf.constant(2);
                var y = tf.constant(5);
                var z = control_flow_ops.cond(tf.less(x, y), () => tf.multiply(x, tf.constant(17)),
                    () => tf.add(y, tf.constant(23)));
                //tf.train.export_meta_graph(@"D:\dev\tensorboard\logdir\sharp.meta", as_text: false);
                self.assertEquals(eval_scalar(z), 34);
            });
        }

        [Ignore("This Test Fails due to missing edges in the graph!")]
        [TestMethod]
        public void testCondFalse()
        {
            with(tf.Graph().as_default(), g =>
            {
                var x = tf.constant(2);
                var y = tf.constant(1);
                var z = control_flow_ops.cond(tf.less(x, y), () => tf.multiply(x, tf.constant(17)),
                    () => tf.add(y, tf.constant(23)));
                self.assertEquals(eval_scalar(z), 24);
            });
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
