using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace TensorFlowNET.UnitTest.control_flow_ops_test
{
    /// <summary>
    /// excerpt of tensorflow/python/framework/ops/control_flow_ops_test.py
    /// </summary>
    [TestClass]
    public class SwitchTestCase : PythonTest
    {

        [Ignore("TODO")]
        [TestMethod]
        public void testResourceReadInLoop()
        {

            //var embedding_matrix = variable_scope.get_variable(
            //"embedding_matrix", initializer: new double[,] { { 2.0 }, { 3.0 } }, use_resource: true);

            /*
            Tensor cond(Tensor it, Tensor _)
            {
                return it < 5;
            }
            */

            // TODO: below code doesn't compile
            //(Tensor, Tensor) body(Tensor it, Tensor cost)
            //{
            //    var embedding = embedding_ops.embedding_lookup(embedding_matrix, new int[]{0});
            //    cost += math_ops.reduce_sum(embedding);
            //    return (it + 1, cost);
            //}
            //var (_, cost1) = control_flow_ops.while_loop(
            //    cond, body, new[]
            //    {
            //        constant_op.constant(0),
            //        constant_op.constant(0.0)
            //    });
            //with<Session>(this.cached_session(), sess =>
            //{
            //    self.evaluate(variables.global_variables_initializer());
            //    self.assertAllEqual(10.0, self.evaluate(cost1));
            //});
        }


        [Ignore("TODO")]
        [TestMethod]
        public void testIndexedSlicesGradientInCondInWhileLoop()
        {
            doTestIndexedSlicesGradientInCondInWhileLoop(use_resource: false);
        }

        [Ignore("TODO")]
        [TestMethod]
        public void testIndexedSlicesGradientInCondInWhileLoopResource()
        {
            doTestIndexedSlicesGradientInCondInWhileLoop(use_resource: true);
        }

        private void doTestIndexedSlicesGradientInCondInWhileLoop(bool use_resource = false)
        {
            //def doTestIndexedSlicesGradientInCondInWhileLoop(self, use_resource=False):
            //  embedding_matrix = variable_scope.get_variable(
            //      "embedding_matrix", [5, 5],
            //      initializer=init_ops.random_normal_initializer(),
            //      use_resource=use_resource)

            //  def cond(it, _):
            //    return it < 5

            //  def body(it, cost):
            //    embedding = embedding_ops.embedding_lookup(embedding_matrix, [0])
            //    cost = control_flow_ops.cond(
            //        math_ops.equal(it, 3), lambda: math_ops.square(cost),
            //        (lambda: cost + math_ops.reduce_sum(embedding)))
            //    return it + 1, cost

            //    _, cost = control_flow_ops.while_loop(
            //        cond, body, [constant_op.constant(0),
            //                     constant_op.constant(0.0)])

            //    dynamic_grads = gradients_impl.gradients(cost, [embedding_matrix])[0]
            //    dynamic_grads = math_ops.segment_sum(dynamic_grads.values,
            //                                         dynamic_grads.indices)

            //    embedding = embedding_ops.embedding_lookup(embedding_matrix, [0])
            //    static = math_ops.square(
            //        math_ops.reduce_sum(embedding) + math_ops.reduce_sum(embedding) +
            //        math_ops.reduce_sum(embedding)) + math_ops.reduce_sum(embedding)
            //    static_grads = gradients_impl.gradients(static, [embedding_matrix])[0]
            //    static_grads = math_ops.segment_sum(static_grads.values,
            //                                        static_grads.indices)

            //    with self.cached_session():
            //      self.evaluate(variables.global_variables_initializer())
            //      self.assertAllEqual(*self.evaluate([static_grads, dynamic_grads]))
        }

        [Ignore("TODO")]
        [TestMethod]
        public void testIndexedSlicesWithShapeGradientInWhileLoop()
        {
            //@test_util.run_v1_only("b/120545219")
            //def testIndexedSlicesWithShapeGradientInWhileLoop(self):
            //  for dtype in [dtypes.float32, dtypes.float64]:
            //    with self.cached_session() as sess:
            //      num_steps = 9

            //      inputs = array_ops.placeholder(dtype=dtype, shape=[num_steps])
            //      initial_outputs = tensor_array_ops.TensorArray(
            //          dtype=dtype, size=num_steps)
            //      initial_i = constant_op.constant(0, dtype=dtypes.int32)

            //      def cond(i, _):
            //        return i < num_steps  # pylint: disable=cell-var-from-loop

            //      def body(i, outputs):
            //        x = array_ops.gather(inputs, i)  # pylint: disable=cell-var-from-loop
            //        outputs = outputs.write(i, x)
            //        return i + 1, outputs

            //      _, outputs = control_flow_ops.while_loop(cond, body,
            //                                               [initial_i, initial_outputs])

            //      outputs = math_ops.reduce_sum(outputs.stack())
            //      r = gradients_impl.gradients([outputs], [inputs])[0]
            //      grad_wr_inputs = ops.convert_to_tensor(r)
            //      o, grad = sess.run([outputs, grad_wr_inputs],
            //                         feed_dict={inputs: [4, 6, 0, 7, 0, 0, 1, 2, 0]})
            //      self.assertEquals(o, 20)
            //      self.assertAllEqual(grad, [1] * num_steps)

        }

        [Ignore("TODO")]
        [TestMethod]
        public void testIndexedSlicesWithDynamicShapeGradientInWhileLoop()
        {
            //@test_util.run_v1_only("b/120545219")
            //def testIndexedSlicesWithDynamicShapeGradientInWhileLoop(self):
            //  for dtype in [dtypes.float32, dtypes.float64]:
            //    with self.cached_session() as sess:
            //      inputs = array_ops.placeholder(dtype=dtype)
            //      initial_outputs = tensor_array_ops.TensorArray(
            //          dtype=dtype, dynamic_size=True, size=1)
            //      initial_i = constant_op.constant(0, dtype=dtypes.int32)

            //      def cond(i, _):
            //        return i < array_ops.size(inputs)  # pylint: disable=cell-var-from-loop

            //      def body(i, outputs):
            //        x = array_ops.gather(inputs, i)  # pylint: disable=cell-var-from-loop
            //        outputs = outputs.write(i, x)
            //        return i + 1, outputs

            //      _, outputs = control_flow_ops.while_loop(cond, body,
            //                                               [initial_i, initial_outputs])

            //      outputs = math_ops.reduce_sum(outputs.stack())
            //      r = gradients_impl.gradients([outputs], [inputs])[0]
            //      grad_wr_inputs = ops.convert_to_tensor(r)
            //      o, grad = sess.run([outputs, grad_wr_inputs],
            //                         feed_dict={inputs: [1, 3, 2]})
            //      self.assertEquals(o, 6)
            //      self.assertAllEqual(grad, [1] * 3)

        }

    }
}
