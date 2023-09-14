using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.ControlFlowTest
{
    [TestClass]
    public class WhileContextTestCase : GraphModeTestBase
    {
        /// <summary>
        /// https://www.tensorflow.org/api_docs/python/tf/while_loop
        /// </summary>
        [TestMethod]
        public void SimpleWhileLoop()
        {
            var i = constant_op.constant(0, name: "i");
            var c = new Func<Tensor, Tensor>(x => tf.less(x, 10, name: "c"));
            var b = new Func<Tensor, Tensor>(x => tf.add(x, 1, name: "c"));
            // var r = control_flow_ops.while_loop(c, b, i);
        }

        private void _testWhileContextHelper(int maximum_iterations)
        {
            // TODO: implement missing code dependencies
            using var sess = this.cached_session();
            var i = constant_op.constant(0, name: "i");
            var c = new Func<Tensor, Tensor>(x => gen_math_ops.less(x, ops.convert_to_tensor(10), name: "c"));
            var b = new Func<Tensor, Tensor>(x => math_ops.add(x, 1, name: "c"));
            //control_flow_ops.while_loop(
            //      c, b, i , maximum_iterations: tf.constant(maximum_iterations));
            foreach (Operation op in sess.graph.get_operations())
            {
                var control_flow_context = op._get_control_flow_context();
                /*if (control_flow_context != null)
                    self.assertProtoEquals(control_flow_context.to_proto(),
                        WhileContext.from_proto(
                            control_flow_context.to_proto()).to_proto(), "");*/
            }
        }

        [Ignore("TODO")]
        [TestMethod]
        public void testWhileContextWithMaximumIterations()
        {
            _testWhileContextHelper(maximum_iterations: 10);
        }
    }
}
