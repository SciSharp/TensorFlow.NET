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

        // NOTE: all other test python test cases of this class are either not needed due to strong typing or dest a deprecated api

    }
}
