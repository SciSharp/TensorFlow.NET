using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow;
using Tensorflow.UnitTest;

namespace TensorFlowNET.UnitTest.ops_test
{
    /// <summary>
    /// excerpt of tensorflow/python/framework/ops_test.py
    /// </summary>
    [TestClass]
    public class GraphTest : GraphModeTestBase
    {
        [TestInitialize]
        public void SetUp()
        {
            ops.reset_default_graph();
        }

        [TestCleanup]
        public void TearDown()
        {
            ops.reset_default_graph();
        }

        private void _AssertDefault(Graph expected)
        {
            Assert.AreSame(ops.get_default_graph(), expected);
        }


        [Ignore("Todo: Port")]
        [TestMethod]
        public void testResetDefaultGraphNesting()
        {
            /*
                      def testResetDefaultGraphNesting(self):
                        g0 = ops.Graph()
                        with self.assertRaises(AssertionError):
                          with g0.as_default():
                            ops.reset_default_graph()
            */
        }

        [Ignore("Todo: Port")]
        [TestMethod]
        public void testGraphContextManagerCancelsEager()
        {
            /*
        def testGraphContextManagerCancelsEager(self):
            with context.eager_mode():
              with ops.Graph().as_default():
                self.assertFalse(context.executing_eagerly())
            */
        }


        [Ignore("Todo: Port")]
        [TestMethod]
        public void testGraphContextManager()
        {
            /*
          def testGraphContextManager(self):
            g0 = ops.Graph()
            with g0.as_default() as g1:
              self.assertIs(g0, g1)
            */
        }

        [Ignore("Todo: Port")]
        [TestMethod]
        public void testDefaultGraph()
        {
            /*
          def testDefaultGraph(self):
            orig = ops.get_default_graph()
            self._AssertDefault(orig)
            g0 = ops.Graph()
            self._AssertDefault(orig)
            context_manager_0 = g0.as_default()
            self._AssertDefault(orig)
            with context_manager_0 as g0:
              self._AssertDefault(g0)
              with ops.Graph().as_default() as g1:
                self._AssertDefault(g1)
              self._AssertDefault(g0)
            self._AssertDefault(orig)
            */
        }

        [Ignore("Todo: Port")]
        [TestMethod]
        public void testPreventFeeding()
        {
            /*
          def testPreventFeeding(self):
            g = ops.Graph()
            a = constant_op.constant(2.0)
            self.assertTrue(g.is_feedable(a))
            g.prevent_feeding(a)
            self.assertFalse(g.is_feedable(a))
            */
        }


        [Ignore("Todo: Port")]
        [TestMethod]
        public void testAsGraphElementConversions()
        {
            /*
        def testAsGraphElementConversions(self):

            class ConvertibleObj(object):

              def _as_graph_element(self):
                return "FloatOutput:0"

            class NonConvertibleObj(object):

              pass

            g = ops.Graph()
            a = _apply_op(g, "FloatOutput", [], [dtypes.float32])
            self.assertEqual(a, g.as_graph_element(ConvertibleObj()))
            with self.assertRaises(TypeError):
              g.as_graph_element(NonConvertibleObj())
            */
        }

        [Ignore("Todo: Port")]
        [TestMethod]
        public void testGarbageCollected()
        {
            /*
          # Regression test against creating custom __del__ functions in classes
          # involved in cyclic references, e.g. Graph and Operation. (Python won't gc
          # cycles that require calling a __del__ method, because the __del__ method can
          # theoretically increase the object's refcount to "save" it from gc, and any
          # already-deleted objects in the cycle would have be to restored.)
          def testGarbageCollected(self):
            # Create a graph we can delete and a weak reference to monitor if it's gc'd
            g = ops.Graph()
            g_ref = weakref.ref(g)
            # Create some ops
            with g.as_default():
              a = constant_op.constant(2.0)
              b = constant_op.constant(3.0)
              c = math_ops.add(a, b)
            # Create a session we can delete
            with session.Session(graph=g) as sess:
              self.evaluate(c)
            # Delete all references and trigger gc
            del g
            del a
            del b
            del c
            del sess
            gc.collect()
            self.assertIsNone(g_ref())
            */
        }

        [Ignore("Todo: Port")]
        [TestMethod]
        public void testRunnableAfterInvalidShape()
        {
            /*
          def testRunnableAfterInvalidShape(self):
            with ops.Graph().as_default():
              with self.assertRaises(ValueError):
                math_ops.add([1, 2], [1, 2, 3])
              a = constant_op.constant(1)
              with session.Session() as sess:
                self.evaluate(a)
            */
        }

        [Ignore("Todo: Port")]
        [TestMethod]
        public void testRunnableAfterInvalidShapeWithKernelLabelMap()
        {
            /*
          def testRunnableAfterInvalidShapeWithKernelLabelMap(self):
            g = ops.Graph()
            with g.as_default():
              with g._kernel_label_map({"KernelLabelRequired": "overload_1"}):
                with self.assertRaises(ValueError):
                  test_ops.kernel_label_required(1)
              a = constant_op.constant(1)
              with session.Session() as sess:
                self.evaluate(a)
            */
        }


    }
}
