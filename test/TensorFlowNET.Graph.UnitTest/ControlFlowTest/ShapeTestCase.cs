using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow;

namespace TensorFlowNET.UnitTest.ControlFlowTest
{
    /// <summary>
    /// excerpt of tensorflow/python/framework/ops/control_flow_ops_test.py
    /// </summary>
    [TestClass]
    public class ShapeTestCase : GraphModeTestBase
    {

        [TestMethod]
        public void testShape()
        {
            var tensor = constant_op.constant(new[] { 1.0, 2.0 });
            self.assertEquals(new long[] { 2 }, tensor.shape.dims);
            self.assertEquals(new long[] { 2 },
                control_flow_ops.with_dependencies(new[] { constant_op.constant(1.0).op }, tensor).shape.dims);
        }

    }
}
