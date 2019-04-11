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

        [Ignore("Todo")]
        [TestMethod]
        public void testCondTrue()
        {
            //var x = constant_op.constant(2);
            //var y = constant_op.constant(5);
            //    var z = control_flow_ops.cond(math_ops.less(x,y), ()=> math_ops.multiply(x, 17), ()=> math_ops.add(y, 23))
            //self.assertEquals(self.evaluate(z), 34);
        }

        [Ignore("Todo")]
        [TestMethod]
        public void testCondFalse()
        {
            //  def testCondFalse(self):
            //    x = constant_op.constant(2)
            //    y = constant_op.constant(1)
            //    z = control_flow_ops.cond(
            //        math_ops.less(
            //            x,
            //            y), lambda: math_ops.multiply(x, 17), lambda: math_ops.add(y, 23))
            //    self.assertEquals(self.evaluate(z), 24)
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
