using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow;

namespace TensorFlowNET.UnitTest
{
    /// <summary>
    /// tensorflow/python/framework/ops_test.py
    /// </summary>
    [TestClass]
    public class ControlDependenciesTest : Python
    {
        [TestMethod]
        public void TestBasic()
        {
            var graph = tf.Graph().as_default();
            Tensor a=null, b = null, c = null, d = null, e = null;
            with<Graph>(graph, g =>
            {
                 a = constant_op.constant(1.0);
                 b = constant_op.constant(1.0);
                with(g.control_dependencies(new ITensorOrOperation[] {a}), x =>
                {
                     c = constant_op.constant(1.0);
                     d = array_ops.identity(b);
                     e = array_ops.identity(c);
                });
            });
            Assert.IsTrue(Enumerable.SequenceEqual(c.op.control_inputs, new[] {a.op}));
            Assert.IsTrue(Enumerable.SequenceEqual(d.op.control_inputs, new[] {a.op}));
            // e should be dominated by c.
            Assert.AreEqual(0, e.op.control_inputs.Length);
        }
    }
}
