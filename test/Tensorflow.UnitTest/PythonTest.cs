using Microsoft.VisualStudio.TestTools.UnitTesting;
using Newtonsoft.Json.Linq;
using Tensorflow.NumPy;
using System.Collections;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest
{
    /// <summary>
    /// Use as base class for test classes to get additional assertions
    /// </summary>
    public class PythonTest
    {
        #region python compatibility layer
        protected PythonTest self { get => this; }
        protected int None => -1;
        #endregion

        #region pytest assertions

        public void assertItemsEqual(ICollection given, ICollection expected)
        {
            if (given is Hashtable && expected is Hashtable)
            {
                Assert.AreEqual(JObject.FromObject(expected).ToString(), JObject.FromObject(given).ToString());
                return;
            }
            Assert.IsNotNull(expected);
            Assert.IsNotNull(given);
            var e = expected.OfType<object>().ToArray();
            var g = given.OfType<object>().ToArray();
            Assert.AreEqual(e.Length, g.Length, $"The collections differ in length expected {e.Length} but got {g.Length}");
            for (int i = 0; i < e.Length; i++)
            {
                /*if (g[i] is NDArray && e[i] is NDArray)
                    assertItemsEqual((g[i] as NDArray).GetData<object>(), (e[i] as NDArray).GetData<object>());
                else*/
                if (e[i] is ICollection && g[i] is ICollection)
                    assertEqual(g[i], e[i]);
                else
                    Assert.AreEqual(e[i], g[i], $"Items differ at index {i}, expected {e[i]} but got {g[i]}");
            }
        }

        public void assertAllEqual(ICollection given, ICollection expected)
        {
            assertItemsEqual(given, expected);
        }

        public void assertFloat32Equal(float expected, float actual, string msg)
        {
            float eps = 1e-6f;
            Assert.IsTrue(Math.Abs(expected - actual) < eps * Math.Max(1.0f, Math.Abs(expected)), $"{msg}: expected {expected} vs actual {actual}");
        }

        public void assertFloat64Equal(double expected, double actual, string msg)
        {
            double eps = 1e-16f;
            Assert.IsTrue(Math.Abs(expected - actual) < eps * Math.Max(1.0f, Math.Abs(expected)), $"{msg}: expected {expected} vs actual {actual}");
        }

        public void AssetSequenceEqual(float[] expected, float[] actual)
        {
            float eps = 1e-5f;
            for (int i = 0; i < expected.Length; i++)
                Assert.IsTrue(Math.Abs(expected[i] - actual[i]) < eps * Math.Max(1.0f, Math.Abs(expected[i])), $"expected {expected} vs actual {actual}");
        }

        public void AssetSequenceEqual(double[] expected, double[] actual)
        {
            double eps = 1e-5f;
            for (int i = 0; i < expected.Length; i++)
                Assert.IsTrue(Math.Abs(expected[i] - actual[i]) < eps * Math.Max(1.0f, Math.Abs(expected[i])), $"expected {expected} vs actual {actual}");
        }

        public void assertEqual(object given, object expected)
        {
            /*if (given is NDArray && expected is NDArray)
            {
                assertItemsEqual((given as NDArray).GetData<object>(), (expected as NDArray).GetData<object>());
                return;
            }*/
            if (given is Hashtable && expected is Hashtable)
            {
                Assert.AreEqual(JObject.FromObject(expected).ToString(), JObject.FromObject(given).ToString());
                return;
            }
            if (given is ICollection collectionGiven && expected is ICollection collectionExpected)
            {
                assertItemsEqual(collectionGiven, collectionExpected);
                return;
            }
            if (given is float && expected is float)
            {
                assertFloat32Equal((float)expected, (float)given, "");
                return;
            }
            if (given is double && expected is double)
            {
                assertFloat64Equal((double)expected, (double)given, "");
                return;
            }
            Assert.AreEqual(expected, given);
        }

        public void assertEquals(object given, object expected)
        {
            assertEqual(given, expected);
        }

        public void assert(object given)
        {
            if (given is bool)
                Assert.IsTrue((bool)given);
            Assert.IsNotNull(given);
        }

        public void assertIsNotNone(object given)
        {
            Assert.IsNotNull(given);
        }

        public void assertFalse(bool cond)
        {
            Assert.IsFalse(cond);
        }

        public void assertTrue(bool cond)
        {
            Assert.IsTrue(cond);
        }

        public void assertAllClose(NDArray array1, NDArray array2, double eps = 1e-5)
        {
            CollectionAssert.AreEqual(array1.ToArray(), array2.ToArray(), new CollectionComparer(eps));

            //TODO: Assert.IsTrue(np.allclose(array1, array2, rtol: eps));
        }

        public void assertAllClose(double value, NDArray array2, double eps = 1e-5)
        {
            if (array2.shape.IsScalar)
            {
                double value2 = array2;
                Assert.AreEqual(value, value2, eps);
                return;
            }
            var array1 = np.ones_like(array2) * value;
            CollectionAssert.AreEqual(array1.ToArray(), array2.ToArray(), new CollectionComparer(eps));

            //TODO: Assert.IsTrue(np.allclose(array1, array2, rtol: eps));
        }

        private class CollectionComparer : IComparer
        {
            private readonly double _epsilon;

            public CollectionComparer(double eps = 1e-06)
            {
                _epsilon = eps;
            }
            public int Compare(object? x, object? y)
            {
                if (x == null && y == null)
                {
                    return 0;
                }
                else if (x == null)
                {
                    return -1;
                }
                else if (y == null)
                {
                    return 1;
                }

                var a = Convert.ToDouble(x);
                var b = Convert.ToDouble(y);

                double delta = Math.Abs(a - b);
                if (delta < _epsilon)
                {
                    return 0;
                }
                return a.CompareTo(b);
            }
        }

        public void assertAllCloseAccordingToType<T>(
            double[,] expected,
            T[,] given,
            double eps = 1e-6,
            float float_eps = 1e-6f)
        {
            Assert.AreEqual(expected.GetLength(0), given.GetLength(0));
            Assert.AreEqual(expected.GetLength(1), given.GetLength(1));

            var flattenGiven = given.Cast<T>().ToArray();
            assertAllCloseAccordingToType(expected, flattenGiven, eps, float_eps);
        }

        public void assertAllCloseAccordingToType<T>(
            ICollection expected,
            ICollection<T> given,
            double eps = 1e-6,
            float float_eps = 1e-6f)
        {
            // TODO: check if any of arguments is not double and change toletance
            // remove givenAsDouble and cast expected instead
            var givenAsDouble = given.Select(x => Convert.ToDouble(x)).ToArray();
            CollectionAssert.AreEqual(expected, givenAsDouble, new CollectionComparer(eps));
        }

        public void assertProtoEquals(object toProto, object o)
        {
            throw new NotImplementedException();
        }

        #endregion

        #region tensor evaluation and test session

        private Session? _cached_session = null;
        private Graph? _cached_graph = null;
        private object? _cached_config = null;
        private bool _cached_force_gpu = false;

        private void _ClearCachedSession()
        {
            if (self._cached_session != null)
            {
                self._cached_session.Dispose();
                self._cached_session = null;
            }
        }

        //protected object _eval_helper(Tensor[] tensors)
        //{
        //    if (tensors == null)
        //        return null;
        //    return nest.map_structure(self._eval_tensor, tensors);
        //}

        protected object? _eval_tensor(object tensor)
        {
            if (tensor == null)
                return None;
            //else if (callable(tensor))
            //     return self._eval_helper(tensor())
            else
            {
                try
                {
                    //TODO:
                    //       if sparse_tensor.is_sparse(tensor):
                    //         return sparse_tensor.SparseTensorValue(tensor.indices, tensor.values,
                    //                                                tensor.dense_shape)
                    //return (tensor as Tensor).numpy();
                }
                catch (Exception)
                {
                    throw new ValueError("Unsupported type: " + tensor.GetType());
                }
                return null;
            }
        }

        /// <summary>
        /// This function is used in many original tensorflow unit tests to evaluate tensors 
        /// in a test session with special settings (for instance constant folding off)
        /// 
        /// </summary>
        public T evaluate<T>(Tensor tensor)
        {
            object? result = null;
            //  if context.executing_eagerly():
            //    return self._eval_helper(tensors)
            //  else:
            {
                var sess = tf.get_default_session();
                var ndarray = tensor.eval(sess);

                if (typeof(T) == typeof(int))
                {
                    int i = ndarray;
                    result = i;
                }
                else if (typeof(T) == typeof(float))
                {
                    float f = ndarray;
                    result = f;
                }
                else if (typeof(T) == typeof(double))
                {
                    double d = ndarray;
                    result = d;
                }
                else if (
                    typeof(T) == typeof(double[])
                    || typeof(T) == typeof(double[,]))
                {
                    result = ndarray.ToMultiDimArray<double>();
                }
                else if (typeof(T) == typeof(float[])
                    || typeof(T) == typeof(float[,]))
                {
                    result = ndarray.ToMultiDimArray<float>();
                }
                else if (typeof(T) == typeof(int[]) 
                    || typeof(T) == typeof(int[,]))
                {
                    result = ndarray.ToMultiDimArray<int>();
                }
                else
                {
                    result = ndarray;
                }

                return (T)result;
            }
        }


        ///Returns a TensorFlow Session for use in executing tests.
        public Session? cached_session(
            Graph? graph = null, object? config = null, bool use_gpu = false, bool force_gpu = false)
        {
            // This method behaves differently than self.session(): for performance reasons
            // `cached_session` will by default reuse the same session within the same
            // test.The session returned by this function will only be closed at the end
            // of the test(in the TearDown function).

            // Use the `use_gpu` and `force_gpu` options to control where ops are run.If
            // `force_gpu` is True, all ops are pinned to `/ device:GPU:0`. Otherwise, if
            // `use_gpu` is True, TensorFlow tries to run as many ops on the GPU as
            // possible.If both `force_gpu and `use_gpu` are False, all ops are pinned to
            // the CPU.

            // Example:
            // python
            // class MyOperatorTest(test_util.TensorFlowTestCase) :
            //   def testMyOperator(self):
            //     with self.cached_session() as sess:
            //       valid_input = [1.0, 2.0, 3.0, 4.0, 5.0]
            //     result = MyOperator(valid_input).eval()
            //       self.assertEqual(result, [1.0, 2.0, 3.0, 5.0, 8.0]
            //       invalid_input = [-1.0, 2.0, 7.0]
            //     with self.assertRaisesOpError("negative input not supported"):
            //         MyOperator(invalid_input).eval()


            // Args:
            //   graph: Optional graph to use during the returned session.
            //   config: An optional config_pb2.ConfigProto to use to configure the
            //     session.
            //   use_gpu: If True, attempt to run as many ops as possible on GPU.
            //   force_gpu: If True, pin all ops to `/device:GPU:0`.

            // Yields:
            //   A Session object that should be used as a context manager to surround
            //   the graph building and execution code in a test case.


            // TODO:
            //  if context.executing_eagerly():
            //    return self._eval_helper(tensors)
            //  else:
            {
                var sess = self._get_cached_session(
                    graph, config, force_gpu, crash_if_inconsistent_args: true);
                using var cached = self._constrain_devices_and_set_default(sess, use_gpu, force_gpu);
                return cached;
            }
        }

        //Returns a TensorFlow Session for use in executing tests.
        public Session session(Graph? graph = null, object? config = null, bool use_gpu = false, bool force_gpu = false)
        {
            //Note that this will set this session and the graph as global defaults.

            //Use the `use_gpu` and `force_gpu` options to control where ops are run.If
            //`force_gpu` is True, all ops are pinned to `/device:GPU:0`. Otherwise, if
            //`use_gpu` is True, TensorFlow tries to run as many ops on the GPU as
            //possible.If both `force_gpu and `use_gpu` are False, all ops are pinned to
            //the CPU.

            //Example:
            //```python
            //class MyOperatorTest(test_util.TensorFlowTestCase):
            //  def testMyOperator(self):
            //    with self.session(use_gpu= True):
            //      valid_input = [1.0, 2.0, 3.0, 4.0, 5.0]
            //    result = MyOperator(valid_input).eval()
            //      self.assertEqual(result, [1.0, 2.0, 3.0, 5.0, 8.0]
            //      invalid_input = [-1.0, 2.0, 7.0]
            //    with self.assertRaisesOpError("negative input not supported"):
            //        MyOperator(invalid_input).eval()
            //```

            //Args:
            //  graph: Optional graph to use during the returned session.
            //  config: An optional config_pb2.ConfigProto to use to configure the
            //    session.
            //  use_gpu: If True, attempt to run as many ops as possible on GPU.
            //  force_gpu: If True, pin all ops to `/device:GPU:0`.

            //Yields:
            //  A Session object that should be used as a context manager to surround
            //  the graph building and execution code in a test case.

            Session? s = null;
            //if (context.executing_eagerly())
            //  yield None
            //else 
            //{
            s = self._create_session(graph, config, force_gpu);
            //}
            return s.as_default();
        }

        private Session? _constrain_devices_and_set_default(Session sess, bool use_gpu, bool force_gpu)
        {
            // Set the session and its graph to global default and constrain devices."""
            if (tf.executing_eagerly())
                return null;
            else
            {
                sess.graph.as_default();
                sess.as_default();
                {
                    if (force_gpu)
                    {
                        // TODO:

                        // Use the name of an actual device if one is detected, or
                        // '/device:GPU:0' otherwise
                        /* var gpu_name = gpu_device_name();
                        if (!gpu_name)
                            gpu_name = "/device:GPU:0"
                        using (sess.graph.device(gpu_name)) {
                            yield return sess;
                        }*/
                        return sess;
                    }
                    else if (use_gpu)
                        return sess;
                    else
                        using (sess.graph.device("/device:CPU:0"))
                            return sess;
                }

            }
        }

        // See session() for details.
        private Session _create_session(Graph? graph, object? cfg, bool forceGpu)
        {
            var prepare_config = new Func<object, object>((config) =>
            {
                //  """Returns a config for sessions.
                //  Args:
                //        config: An optional config_pb2.ConfigProto to use to configure the
                //      session.
                //  Returns:
                //    A config_pb2.ConfigProto object.

                //TODO: config

                //  # use_gpu=False. Currently many tests rely on the fact that any device
                //  # will be used even when a specific device is supposed to be used.
                //  allow_soft_placement = not force_gpu
                //  if config is None:
                //    config = config_pb2.ConfigProto()
                //    config.allow_soft_placement = allow_soft_placement
                //    config.gpu_options.per_process_gpu_memory_fraction = 0.3
                //  elif not allow_soft_placement and config.allow_soft_placement:
                //    config_copy = config_pb2.ConfigProto()
                //    config_copy.CopyFrom(config)
                //    config = config_copy
                //    config.allow_soft_placement = False
                //  # Don't perform optimizations for tests so we don't inadvertently run
                //  # gpu ops on cpu
                //  config.graph_options.optimizer_options.opt_level = -1
                //  # Disable Grappler constant folding since some tests & benchmarks
                //  # use constant input and become meaningless after constant folding.
                //  # DO NOT DISABLE GRAPPLER OPTIMIZERS WITHOUT CONSULTING WITH THE
                //  # GRAPPLER TEAM.
                //  config.graph_options.rewrite_options.constant_folding = (
                //      rewriter_config_pb2.RewriterConfig.OFF)
                //  config.graph_options.rewrite_options.pin_to_host_optimization = (
                //      rewriter_config_pb2.RewriterConfig.OFF)
                return config;
            });
            //TODO: use this instead of normal session
            //return new ErrorLoggingSession(graph = graph, config = prepare_config(config))
            return new Session(graph);//, config = prepare_config(config))
        }

        private Session _get_cached_session(
                          Graph? graph = null,
                          object? config = null,
                          bool force_gpu = false,
                          bool crash_if_inconsistent_args = true)
        {
            // See cached_session() for documentation.
            if (self._cached_session == null)
            {
                var sess = self._create_session(graph, config, force_gpu);
                self._cached_session = sess;
                self._cached_graph = graph;
                self._cached_config = config;
                self._cached_force_gpu = force_gpu;
                return sess;
            }
            else
            {

                if (crash_if_inconsistent_args && self._cached_graph != null && !self._cached_graph.Equals(graph))
                    throw new ValueError(@"The graph used to get the cached session is 
                                           different than the one that was used to create the
                                           session. Maybe create a new session with 
                                           self.session()");
                if (crash_if_inconsistent_args && self._cached_config != null && !self._cached_config.Equals(config))
                {
                    throw new ValueError(@"The config used to get the cached session is 
                                           different than the one that was used to create the 
                                           session. Maybe create a new session with 
                                           self.session()");
                }
                if (crash_if_inconsistent_args && !self._cached_force_gpu.Equals(force_gpu))
                {
                    throw new ValueError(@"The force_gpu value used to get the cached session is 
                                           different than the one that was used to create the 
                                           session. Maybe create a new session with 
                                           self.session()");
                }
                return self._cached_session;
            }
        }

        [TestCleanup]
        public void Cleanup()
        {
            _ClearCachedSession();
        }

        #endregion

        public void AssetSequenceEqual<T>(T[] a, T[] b)
        {
            Assert.IsTrue(Enumerable.SequenceEqual(a, b));
        }
    }
}
