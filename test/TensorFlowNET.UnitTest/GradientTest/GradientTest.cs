using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow;
using Tensorflow.UnitTest;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.Gradient
{
    [TestClass]
    public class GradientTest : GraphModeTestBase
    {
        [TestMethod]
        public void BroadcastToGrad()
        {
            var graph = tf.Graph().as_default();

            var x = tf.constant(2, dtype: dtypes.float32);
            var y = tf.broadcast_to(x, (2, 4, 3));
            var grad = tf.gradients(y, x);

            using (var sess = tf.Session(graph))
            {
                float result = sess.run(grad[0]);
                Assert.AreEqual(result, 24.0f);
            }
        }

        [TestMethod]
        public void CumsumGrad()
        {
            var graph = tf.Graph().as_default();

            var x = tf.constant(2, dtype: dtypes.float32);
            var y = tf.broadcast_to(x, (2, 4, 3));
            var z = tf.cumsum(y, axis: 1);
            var grad = tf.gradients(z, x);

            using (var sess = tf.Session(graph))
            {
                float result = sess.run(grad[0]);
                Assert.AreEqual(result, 60.0f);
            }
        }

        [TestMethod, Ignore]
        public void testGradients()
        {
            var g = tf.Graph().as_default();
            var inp = tf.constant(1.0, shape: new[] { 32, 100 }, name: "in");
            var w = tf.constant(1.0, shape: new[] { 100, 10 }, name: "w");
            var b = tf.Variable(1.0, shape: new[] { 10 }, name: "b");
            var xw = math_ops.matmul(inp, w, name: "xw");
            var h = nn_ops.bias_add(xw, b, name: "h");
            var w_grad = gradients_impl.gradients(new[] { h }, new[] { w })[0];
            self.assertEquals("MatMul", w_grad.op.type);
            // TODO: Operation._original_op
            //self.assertEquals(w_grad.op._original_op, xw.op);
            self.assertTrue((bool)w_grad.op.get_attr("transpose_a"));
            self.assertFalse((bool)w_grad.op.get_attr("transpose_b"));
        }

        [TestMethod]
        public void testBatchMatMulGradient()
        {
            var a = tf.constant(np.array(Enumerable.Range(1, 18).Select(elem => (float)elem).ToArray()), shape: new[] { 2, 3, 3 });
            var b = tf.divide(a, tf.constant(2.0f));
            var c = tf.batch_matmul(a, b);
            var g = tf.gradients(c, new[] { a, b }, stop_gradients: new[] { a, b });
            var checkG = new[]
            {
                3.0f, 7.5f, 12.0f,
                3.0f, 7.5f, 12.0f,
                3.0f, 7.5f, 12.0f,
                16.5f, 21.0f, 25.5f,
                16.5f, 21.0f, 25.5f,
                16.5f, 21.0f, 25.5f,
                12.0f, 12.0f, 12.0f,
                15.0f, 15.0f, 15.0f,
                18.0f, 18.0f, 18.0f,
                39.0f, 39.0f, 39.0f,
                42.0f, 42.0f, 42.0f,
                45.0f, 45.0f, 45.0f
            };
            using (var sess = tf.Session())
            {
                var result = sess.run(g);
                var resultList = result[0].GetData<float>().ToList();
                resultList.AddRange(result[1].GetData<float>());
                Console.WriteLine(result.ToString());
                CollectionAssert.AreEqual(resultList.ToArray(), checkG);
            }
        }

        [TestMethod]
        public void testSimpleGradients()
        {
            (T, T) evaluateDerivatives<T>(Func<Tensor, Tensor> f, T xval) where T : unmanaged
            {
                var x = tf.constant(xval);
                var y = f(x);
                var g = tf.gradients(y, x);

                using (var session = tf.Session())
                {
                    var result = session.run(new[] { y, g[0] });
                    return (result[0].GetData<T>()[0], result[1].GetData<T>()[0]);
                }
            }

            void test(string name, Func<Tensor, Tensor> tfF, Func<double, (double, double)> targetF, double[] values)
            {
                foreach (var x in values)
                {
                    var (expectedY, expectedDY) = targetF(x);

                    {
                        var (actualY, actualDY) = evaluateDerivatives(tfF, x);
                        self.assertFloat64Equal(expectedY, actualY, $"value {name}/float64 at {x}");
                        self.assertFloat64Equal(expectedDY, actualDY, $"derivative {name}/float64 at {x}");
                    }

                    {
                        var (actualY, actualDY) = evaluateDerivatives(tfF, (float)x);
                        self.assertFloat32Equal((float)expectedY, actualY, $"value {name}/float32 at {x}");
                        self.assertFloat32Equal((float)expectedDY, actualDY, $"derivative {name}/float32 at {x}");
                    }
                }
            }

            test("tf.exp",
                x => tf.exp(5 * x),
                x => (Math.Exp(5.0 * x), 5.0 * Math.Exp(5.0 * x)),
                new[] { -1.0, 0.0, 1.0, 1.5 });

            test("tf.log",
                x => tf.log(x),
                x => (Math.Log(x), 1.0 / x),
                new[] { 0.5, 1.0, 1.5, 2.0 });

            test("tf.sqrt",
                x => tf.sqrt(x),
                x => (Math.Sqrt(x), 0.5 / Math.Sqrt(x)),
                new[] { 0.5, 1.0, 1.1, 1.5, 2.0 });

            test("tf.sin",
                x => tf.sin(x),
                x => (Math.Sin(x), Math.Cos(x)),
                new[] { -1.0, 0.0, 1.0, 1.5, 2.0 });

            test("tf.sinh",
                x => tf.sinh(x),
                x => (Math.Sinh(x), Math.Cosh(x)),
                new[] { -1.0, 0.0, 1.0, 1.5, 2.0 });

            test("tf.cos",
                x => tf.cos(x),
                x => (Math.Cos(x), -Math.Sin(x)),
                new[] { -1.0, 0.0, 1.0, 1.5, 2.0 });

            test("tf.cosh",
                x => tf.cosh(x),
                x => (Math.Cosh(x), Math.Sinh(x)),
                new[] { -1.0, 0.0, 1.0, 1.5, 2.0 });

            test("tf.tanh",
                x => tf.tanh(x),
                x => (Math.Tanh(x), 1.0 - Math.Pow(Math.Tanh(x), 2.0)),
                new[] { -1.0, 0.0, 1.0, 1.5, 2.0 });

            test("tf.maximum",
                x => tf.maximum(x, tf.constant(0.0, dtype: x.dtype)),
                x => (Math.Max(x, 0.0), (x > 0.0) ? 1.0 : 0.0),
                new[] { -1.0, 1.0 });

            test("tf.minimum",
                x => tf.minimum(x, tf.constant(0.0, dtype: x.dtype)),
                x => (Math.Min(x, 0.0), (x < 0.0) ? 1.0 : 0.0),
                new[] { -1.0, 1.0 });
        }

        [TestMethod]
        public void testTanhGradient()
        {
            var a = tf.constant(1f);
            var b = tf.tanh(a);
            var g = tf.gradients(b, a);
            using (var sess = tf.Session())
            {
                var result = sess.run(g);
                var actual = result[0].GetData<float>()[0];
                self.assertEquals(0.41997434127f, actual);
            }
        }


        [TestMethod]
        public void testLgammaGrad()
        {
            var a = tf.constant(5f);
            var b = tf.lgamma(a);
            var g = tf.gradients(b, a);
            using (var sess = tf.Session())
            {
                var result = sess.run(new object[] { g, b });
                var actualDeriv = result[0].GetData<float>()[0];
                var actual = result[1].GetData<float>()[0];
                self.assertEquals(1.5061177f, actualDeriv);
                self.assertEquals(3.17805386f, actual);
            }
        }

        [TestMethod]
        public void testSliceGrad()
        {
            var a = tf.tanh(tf.constant(new[] { 2f, 3f }, shape: new[] { 2, 1 }));
            var b = tf.strided_slice(a,
                tf.constant(new[] { 0 }, tf.int32, new[] { 1 }),
                tf.constant(new[] { 1 }, tf.int32, new[] { 1 }),
                tf.constant(new[] { 1 }, tf.int32, new[] { 1 })
            );
            var g = tf.gradients(b, a);
            using (var sess = tf.Session())
            {
                var result = sess.run(new object[] { g, b });
                var actualDeriv = np.squeeze(result[0]);
                var actual = np.squeeze(result[1]);
                self.assertEquals(new float[] { 1, 0 }, new float[] { actualDeriv[0], actualDeriv[1] });
                self.assertEquals(0.9640276f, (float)actual);
            }
        }

        [TestMethod]
        public void testConcatGrad()
        {
            var a1 = tf.constant(new[] { 2f }, shape: new[] { 1 });
            var a2 = tf.constant(new[] { 3f }, shape: new[] { 1 });
            var a = tf.concat(new List<Tensor>(new[] { a1, a2 }), 0);
            var g = tf.gradients(a, a1);
            using (var sess = tf.Session())
            {
                var result = sess.run(new object[] { g, a });
                var actualDeriv = result[0].GetData<float>()[0];
                var actual = result[1].GetData<float>()[0];
                self.assertEquals(1f, actualDeriv);
                self.assertEquals(2f, actual);
            }
        }

        [TestMethod]
        public void testStopGradientFunction()
        {
            var ap = tf.constant(1f);
            var b = tf.tanh(ap) + gen_array_ops.stop_gradient(ap);
            var g = tf.gradients(b, ap);
            using (var sess = tf.Session())
            {
                var result = sess.run(g);
                var actual = result[0].GetData<float>()[0];
                self.assertEquals(0.41997434127f, actual);
            }
        }
        [Ignore("TODO")]
        [TestMethod]
        public void testUnusedOutput()
        {
            //def testUnusedOutput(self):
            //  with ops.Graph().as_default():
            //    w = constant(1.0, shape=[2, 2])
            //    x = constant(1.0, shape=[2, 2])
            //    wx = math_ops.matmul(w, x)
            //    split_wx = array_ops.split(value=wx, num_or_size_splits=2, axis=0)
            //    c = math_ops.reduce_sum(split_wx[1])
            //    gw = gradients.gradients(c, [w])[0]
            //  self.assertEquals("MatMul", gw.op.type)
        }

        [Ignore("TODO")]
        [TestMethod]
        public void testColocateGradients()
        {

            //def testColocateGradients(self):
            //  with ops.Graph().as_default() as g:
            //    w = constant(1.0, shape=[1, 1])
            //    x = constant(1.0, shape=[1, 2])
            //    with g.device("/device:GPU:0"):
            //      wx = math_ops.matmul(w, x)
            //    gw = gradients.gradients(wx, [w], colocate_gradients_with_ops=True)[0]
            //  self.assertEqual(gw.op.colocation_groups(), wx.op.colocation_groups())
        }

        [Ignore("TODO")]
        [TestMethod]
        public void testColocateGradientsWithAggregation()
        {
            //def testColocateGradientsWithAggregation(self):
            //  with ops.Graph().as_default() as g:
            //    with g.device("/device:GPU:1"):
            //      w = constant(1.0, shape=[1, 1])
            //    x = constant(1.0, shape=[1, 2])
            //    y = constant(1.0, shape=[1, 2])
            //    wx = math_ops.matmul(w, x)
            //    wy = math_ops.matmul(w, y)
            //    with g.device("/device:GPU:0"):
            //      z = wx + wy

            //    gw1 = gradients.gradients(z, [w], colocate_gradients_with_ops=True)[0]
            //    self.assertEqual(gw1.op.colocation_groups(), wx.op.colocation_groups())

            //    gw2 = gradients.gradients(z, [w], colocate_gradients_with_ops=False)[0]
            //    self.assertTrue(wx.op.colocation_groups() != gw2.op.colocation_groups())

        }

        [Ignore("TODO")]
        [TestMethod]
        public void testColocateGradientsWithAggregationInMultipleDevices()
        {
            //def testColocateGradientsWithAggregationInMultipleDevices(self):
            //  with ops.Graph().as_default() as g:
            //    with g.device("/device:GPU:1"):
            //      w = constant(1.0, shape=[1, 1])
            //    x = constant(1.0, shape=[1, 2])
            //    y = constant(1.0, shape=[1, 2])
            //    with g.device("/task:1"):
            //      wx = math_ops.matmul(w, x)
            //    with g.device("/task:2"):
            //      wy = math_ops.matmul(w, y)
            //    with g.device("/device:GPU:0"):
            //      z = wx + wy

            //    gw1 = gradients.gradients(z, [w], colocate_gradients_with_ops=True)[0]
            //    self.assertEqual(gw1.op.colocation_groups(), w.op.colocation_groups())

            //    gw2 = gradients.gradients(z, [w], colocate_gradients_with_ops=False)[0]
            //    self.assertTrue(w.op.colocation_groups() != gw2.op.colocation_groups())
        }


        [Ignore("TODO")]
        [TestMethod]
        public void testColocateGradientsWithGateGradients()
        {

            //def testColocateGradientsWithGateGradients(self):
            //  if not test_util.is_gpu_available():
            //    self.skipTest("No GPU available")
            //  with ops.Graph().as_default() as g:
            //    with g.device("/device:CPU:0"):
            //      x = constant(1.0, shape=[1, 1])
            //      y = constant(1.0, shape=[1, 1])
            //      s = x + y
            //    with g.device("/device:GPU:0"):
            //      z = math_ops.reduce_sum(s)

            //    gz_x = gradients.gradients(z, [x], colocate_gradients_with_ops=True,
            //                               gate_gradients=True)[0]
            //    with session.Session():
            //      # Make sure the placer doesn't complain.
            //      self.evaluate(gz_x)

        }

        [Ignore("TODO")]
        [TestMethod]
        public void testBoundaryStop()
        {
            //def testBoundaryStop(self):
            //  # Test that we don't differentiate 'x'. The gradient function for 'x' is
            //  # set explicitly to None so we will get an exception if the gradient code
            //  # tries to differentiate 'x'.
            //  with ops.Graph().as_default():
            //    c = constant(1.0)
            //    x = array_ops.identity(c)
            //    y = x + 1.0
            //    z = y + 1
            //    grads = gradients.gradients(z, [x])
            //    self.assertTrue(all(x is not None for x in grads))

        }

        [Ignore("TODO")]
        [TestMethod]
        public void testBoundaryContinue()
        {
            //@test_util.run_v1_only("b/120545219")
            //def testBoundaryContinue(self):
            //  # Test that we differentiate both 'x' and 'y' correctly when x is a
            //  # predecessor of y.
            //  with self.cached_session():
            //    x = constant(1.0)
            //    y = x * 2.0
            //    z = y * 3.0
            //    grads = gradients.gradients(z, [x, y])
            //    self.assertTrue(all(x is not None for x in grads))
            //    self.assertEqual(6.0, grads[0].eval())

        }

        [Ignore("TODO")]
        [TestMethod]
        public void testAggregationMethodAccumulateN()
        {

            //@test_util.run_v1_only("b/120545219")
            //def testAggregationMethodAccumulateN(self):
            //  with self.cached_session():
            //    x = constant(1.0)
            //    y = x * 2.0
            //    z = y + y + y + y + y + y + y + y + y + y
            //    grads = gradients.gradients(
            //        z, [x, y],
            //        aggregation_method=gradients.AggregationMethod.
            //        EXPERIMENTAL_ACCUMULATE_N)
            //    self.assertTrue(all(x is not None for x in grads))
            //    self.assertEqual(20.0, grads[0].eval())
            //    self.assertEqual(10.0, grads[1].eval())

        }

        [Ignore("TODO")]
        [TestMethod]
        public void testAggregationMethodAddN()
        {
            //@test_util.run_v1_only("b/120545219")
            //def testAggregationMethodAddN(self):
            //  with self.cached_session():
            //    x = constant(1.0)
            //    y = x * 2.0
            //    z = y + y + y + y + y + y + y + y + y + y
            //    grads = gradients.gradients(
            //        z, [x, y], aggregation_method=gradients.AggregationMethod.ADD_N)
            //    self.assertTrue(all(x is not None for x in grads))
            //    self.assertEqual(20.0, grads[0].eval())
            //    self.assertEqual(10.0, grads[1].eval())


        }

        [Ignore("TODO")]
        [TestMethod]
        public void testAggregationMethodTree()
        {
            //@test_util.run_v1_only("b/120545219")
            //def testAggregationMethodTree(self):
            //  with self.cached_session():
            //    x = constant(1.0)
            //    y = x * 2.0
            //    z = y + y + y + y + y + y + y + y + y + y
            //    grads = gradients.gradients(
            //        z, [x, y],
            //        aggregation_method=gradients.AggregationMethod.EXPERIMENTAL_TREE)
            //    self.assertTrue(all(x is not None for x in grads))
            //    self.assertEqual(20.0, grads[0].eval())
            //    self.assertEqual(10.0, grads[1].eval())

        }

        [Ignore("TODO")]
        [TestMethod]
        public void testNoGradientForStringOutputs()
        {

            //def testNoGradientForStringOutputs(self):
            //  with ops.Graph().as_default():

            //    def _TestOpGrad(_, float_grad, string_grad):
            //      """Gradient function for TestStringOutput."""
            //      self.assertEquals(float_grad.dtype, dtypes.float32)
            //      self.assertFalse(string_grad)
            //      return float_grad

            //    ops.RegisterGradient("TestStringOutput")(_TestOpGrad)

            //    c = constant(1.0)
            //    x, _ = test_ops.test_string_output(c)
            //    z = x * 2.0
            //    w = z * 3.0
            //    grads = gradients.gradients(z, [c])
            //    self.assertTrue(isinstance(grads[0], ops.Tensor))
            //    grads = gradients.gradients(w, [c])
            //    self.assertTrue(isinstance(grads[0], ops.Tensor))
        }

        [Ignore("TODO")]
        [TestMethod]
        public void testSingletonIndexedSlices()
        {

            //def testSingletonIndexedSlices(self):
            //  with ops.Graph().as_default():
            //    x = array_ops.placeholder(dtypes.float32)
            //    y = array_ops.identity(x)
            //    dy = ops.IndexedSlices(
            //        array_ops.placeholder(dtypes.float32),
            //        array_ops.placeholder(dtypes.int32))
            //    dx, = gradients.gradients(y, x, grad_ys=dy)
            //    # The IndexedSlices gradient of tf.identity is the identity map.
            //    with self.cached_session() as sess:
            //      vdx, vdy = sess.run(
            //          [dx, dy], feed_dict={x: [1.0], dy.indices: [0], dy.values: [2.0]})
            //    self.assertEqual(vdx, vdy)
        }

        [Ignore("TODO")]
        [TestMethod]
        public void testNonDifferentiableSwitchInWhileLoop()
        {


            //@test_util.run_v1_only("b/120545219")
            //def testNonDifferentiableSwitchInWhileLoop(self):
            //  with ops.Graph().as_default():
            //    v = array_ops.placeholder(dtypes.float32, [])

            //    def _Step(i, a, ta):
            //      a += math_ops.cast(v, dtypes.int32)
            //      return (i + 1, a, ta.write(i, a))

            //    n = 4
            //    i, _, ta = control_flow_ops.while_loop(
            //        lambda i, *_: i < n,
            //        _Step, [0, 0, tensor_array_ops.TensorArray(
            //            dtypes.int32, size=n)])
            //    target = ta.read(i - 1)
            //    grad, = gradients.gradients(target, v)
            //    self.assertIsNone(grad)

        }

        [Ignore("TODO")]
        [TestMethod]
        public void testVariableReadValueGradient()
        {

            //def testVariableReadValueGradient(self):
            //  with ops.Graph().as_default():
            //    init = constant_op.constant(100.0)
            //    var = variables.Variable(init)
            //    gradient = gradients.gradients(var.read_value(), var)
            //    self.assertIsNotNone(gradient)
        }

        [Ignore("TODO")]
        [TestMethod]
        public void testVariableAsGraphElementGradient()
        {
            //def testVariableAsGraphElementGradient(self):
            //  with ops.Graph().as_default() as graph:
            //    init = constant_op.constant(100.0)
            //    var = variables.Variable(init)
            //    gradient = gradients.gradients(graph.as_graph_element(var), var)
            //    self.assertIsNotNone(gradient)
        }

        [Ignore("TODO")]
        [TestMethod]
        public void testVariableRefGradient()
        {

            //@test_util.run_v1_only("b/120545219")
            //def testVariableRefGradient(self):
            //  with ops.Graph().as_default():
            //    init = constant_op.constant(100.0)
            //    var = variables.VariableV1(init)
            //    gradient = gradients.gradients(var._ref(), var)
            //    self.assertIsNotNone(gradient)
        }

        [Ignore("TODO")]
        [TestMethod]
        public void testDependentYs()
        {
            //@test_util.run_v1_only("b/120545219")
            //def testDependentYs(self):
            //  with self.cached_session():
            //    x = constant_op.constant(3.0)
            //    y = math_ops.square(x)
            //    y1 = math_ops.square(y)
            //    y2 = math_ops.square(y1)
            //    g = gradients.gradients([y, y2], x)
            //    self.assertAllClose(17502.0, g[0].eval())
            //    g = gradients.gradients(y + y2, x)
            //    self.assertAllClose(17502.0, g[0].eval())
            //    z = array_ops.identity(y)
            //    z2 = array_ops.identity(y2)
            //    g = gradients.gradients([z, z2], x)
            //    self.assertAllClose(17502.0, g[0].eval())

        }

        [Ignore("TODO")]
        [TestMethod]
        public void testPartialDerivatives()
        {

            //@test_util.run_v1_only("b/120545219")
            //def testPartialDerivatives(self):
            //  with self.cached_session():
            //    x = constant_op.constant(1.)
            //    y = 2 * x
            //    z = x + y
            //    totalg = gradients.gradients(z, [x, y])
            //    self.assertEqual([3.0, 1.0], [g.eval() for g in totalg])
            //    partialg = gradients.gradients(z, [x, y], stop_gradients=[x, y])
            //    self.assertEqual([1.0, 1.0], [g.eval() for g in partialg])
        }

        [Ignore("TODO")]
        [TestMethod]
        public void testStopGradients()
        {


            //@test_util.run_v1_only("b/120545219")
            //def testStopGradients(self):
            //  def _MakeGraph(rng, stop_gradients=()):
            //    def _FunctionOf(xs, k=3):
            //      return ops.convert_to_tensor(
            //          sum(math_ops.matmul(rng.rand(k, k), x) for x in xs)
            //          + rng.rand(k, k))

            //    a = _FunctionOf([])
            //    if "a" in stop_gradients: a = array_ops.stop_gradient(a)
            //    b = _FunctionOf([a])
            //    if "b" in stop_gradients: b = array_ops.stop_gradient(b)
            //    c = _FunctionOf([a, b])
            //    if "c" in stop_gradients: c = array_ops.stop_gradient(c)
            //    d = _FunctionOf([b, c])
            //    if "d" in stop_gradients: d = array_ops.stop_gradient(d)
            //    return dict(a=a, b=b, c=c, d=d)

            //  def _Gradients(ys, xs, **kwargs):
            //    dydxs = gradients.gradients(ys, xs, **kwargs)
            //    dydxs = [0. * x if dydx is None else dydx
            //             for x, dydx in zip(xs, dydxs)]
            //    return dydxs
            //  seed = np.random.randint(1000)
            //  cases = []
            //  subsets = [""] + "a b c d ab ac ad bc bd cd abc abd acd bcd abcd".split()
            //  graph = _MakeGraph(np.random.RandomState(seed))
            //  for constants in subsets:
            //    graph_with_stops = _MakeGraph(np.random.RandomState(seed), constants)
            //    for variables_ in subsets:
            //      # compute the gradient when stopped using tf.stop_gradients
            //      grad1 = _Gradients([graph_with_stops["d"]],
            //                         [graph_with_stops[v] for v in variables_])
            //      # compute the gradient when stopped using the stop_gradients kwarg
            //      grad2 = _Gradients([graph["d"]],
            //                         [graph[v] for v in variables_],
            //                         stop_gradients=[graph[v] for v in constants])
            //      cases.append(dict(grad1=grad1, grad2=grad2,
            //                        constants=constants, variables=variables_))

            //  # evaluate all tensors in one call to session.run for speed
            //  with self.cached_session() as sess:
            //    results = sess.run([(case["grad1"], case["grad2"]) for case in cases])

            //  for (npgrad1, npgrad2), case in zip(results, cases):
            //    for a, b in zip(npgrad1, npgrad2):
            //      np.testing.assert_allclose(a, b)

        }

        [Ignore("TODO")]
        [TestMethod]
        public void testUnconnectedGradientsNoneUnconnectedGradients()
        {


            //def testUnconnectedGradientsNoneUnconnectedGradients(self):
            //  with ops.Graph().as_default():
            //    x = constant(1.0, shape=[2, 2])
            //    y = constant(3.0, shape=[3, 1])
            //    grad = gradients.gradients(
            //        [y], [x], unconnected_gradients="none")
            //  self.assertIsNone(grad[0])
        }

        [Ignore("TODO")]
        [TestMethod]
        public void testUnconnectedGradientsZerosUnconnectedGradients()
        {


            //def testUnconnectedGradientsZerosUnconnectedGradients(self):
            //  with ops.Graph().as_default():
            //    x = constant(1.0, shape=[2, 2])
            //    y = constant(3.0, shape=[3, 1])
            //    grads = gradients.gradients(
            //        [y], [x], unconnected_gradients="zero")
            //    with self.cached_session() as sess:
            //      self.assertAllEqual([[0.0, 0.0], [0.0, 0.0]], self.evaluate(grads)[0])
        }

        [Ignore("TODO")]
        [TestMethod]
        public void testUnconnectedGradientsZeroConnectedGradients()
        {



            //def testUnconnectedGradientsZeroConnectedGradients(self):
            //  with ops.Graph().as_default():
            //    x = constant(1.0)
            //    y = x * 3.0
            //    grad = gradients.gradients(
            //        [y], [x], unconnected_gradients="zero")
            //    with self.cached_session() as sess:
            //      self.assertEquals(3.0, self.evaluate(grad)[0])
        }

        [Ignore("TODO")]
        [TestMethod]
        public void testUnknownUnconnectedGradientsValueGiven()
        {
            //def testUnknownUnconnectedGradientsValueGiven(self):
            //  with ops.Graph().as_default():
            //    x = constant(1.0)
            //    y = constant(1.0)
            //    with self.assertRaisesRegexp(
            //        ValueError, "Unknown value for unconnected_gradients: 'nonsense'"):
            //      gradients.gradients([y], [x], unconnected_gradients="nonsense")

        }



        /*



         */
    }
}
