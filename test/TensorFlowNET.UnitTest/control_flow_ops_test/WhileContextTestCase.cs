using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow;
using static Tensorflow.Python;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.control_flow_ops_test
{
    [TestClass]
    public class WhileContextTestCase : PythonTest
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
            var r = control_flow_ops.while_loop(c, b, new[] { i });
        }

        private void _testWhileContextHelper(int? maximum_iterations = null)
        {
            // TODO: implement missing code dependencies
            using (var sess = this.cached_session())
            {
                var i = constant_op.constant(0, name: "i");
                var c = new Func<Tensor, Tensor>(x => gen_math_ops.less(x, 10, name: "c"));
                var b = new Func<Tensor, Tensor>(x => gen_math_ops.add(x, 1, name: "c"));
                control_flow_ops.while_loop(
                      c, b, new[] { i }, maximum_iterations: maximum_iterations);
                foreach (Operation op in sess.graph.get_operations())
                {
                    var control_flow_context = op._get_control_flow_context();
                    /*if (control_flow_context != null)
                        self.assertProtoEquals(control_flow_context.to_proto(),
                            WhileContext.from_proto(
                                control_flow_context.to_proto()).to_proto(), "");*/
                }
            }
        }

        [Ignore("TODO")]
        [TestMethod]
        public void testWhileContext()
        {
            _testWhileContextHelper();
        }

        [Ignore("TODO")]
        [TestMethod]
        public void testWhileContextWithMaximumIterations()
        {
            _testWhileContextHelper(maximum_iterations: 10);
        }
    }
}
