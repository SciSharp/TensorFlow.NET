using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow;
using Tensorflow.Eager;

namespace TensorFlowNET.UnitTest
{
    /// <summary>
    /// excerpt of tensorflow/python/framework/ops_test.py
    /// </summary>
    [TestClass]
    public class ControlDependenciesTest : PythonTest
    {
        [TestMethod]
        public void TestBasic()
        {
            var graph = tf.Graph().as_default();
            Tensor a = null, b = null, c = null, d = null, e = null;
            with<Graph>(graph, g =>
            {
                a = constant_op.constant(1.0);
                b = constant_op.constant(1.0);
                with(g.control_dependencies(new ITensorOrOperation[] { a }), x =>
                  {
                      c = constant_op.constant(1.0);
                      d = array_ops.identity(b);
                      e = array_ops.identity(c);
                  });
            });
            Assert.IsTrue(Enumerable.SequenceEqual(c.op.control_inputs, new[] { a.op }));
            Assert.IsTrue(Enumerable.SequenceEqual(d.op.control_inputs, new[] { a.op }));
            // e should be dominated by c.
            Assert.AreEqual(0, e.op.control_inputs.Length);
        }

        [Ignore("Part of this test is not compiling")]
        [TestMethod]
        public void TestEager()
        {
            Tensor a = null, b = null, c = null, d = null, e = null;
            var calls = 0;
            Func<Tensor> future = () =>
            {

                calls += 1;
                return constant_op.constant(2.0);
            };
            using (var opts = new ContextOptions())
            using (var status = new Status())
            using (var context = new Context(opts, status))
            {
                if (context.executing_eagerly())
                {
                    // TODO: make this compile (see original Python code below)
                    //a = constant_op.constant(1.0);
                    //b = future; // <--- {henon} obviously, this doesn't compile, looks like control_dependencies needs to be able to take callables as well. 
                    //with(ops.control_dependencies(new Operation[] {a, b}), ctrl =>
                    //{
                    //    return c = constant_op.constant(3.0);
                    //});
                    //Assert.AreEqual(calls, 1);
                }
                else
                {
                    var graph = tf.Graph().as_default();
                    with<Graph>(graph, g =>
                    {
                        a = constant_op.constant(1.0);
                        b = future();
                        with(g.control_dependencies(new ITensorOrOperation[] { a, b }), ctrl =>
                          {
                              c = constant_op.constant(3.0);
                          });
                        Assert.IsTrue(Enumerable.SequenceEqual(c.op.control_inputs, new[] { a.op, b.op }));
                        Assert.AreEqual(1, calls);
                    });

                }
            }
            /*
              def testEager(self):
                def future():
                  future.calls += 1
                  return constant_op.constant(2.0)
                future.calls = 0

                if context.executing_eagerly():
                  a = constant_op.constant(1.0)
                  b = future
                  with ops.control_dependencies([a, b]):
                    c = constant_op.constant(3.0)
                  self.assertEqual(future.calls, 1)
                else:
                  g = ops.Graph()
                  with g.as_default():
                    a = constant_op.constant(1.0)
                    b = future()
                    with g.control_dependencies([a, b]):
                      c = constant_op.constant(3.0)
                  self.assertEqual(c.op.control_inputs, [a.op, b.op])
                  self.assertEqual(future.calls, 1)
            */
        }


        // Note: {henon}, all tests below use the function _apply_op which is not really portable in C#, see original source below
        // but I think _apply_op(...) can just be replaced by g.create_op(...).
        /*
def _apply_op(g, *args, **kwargs):
  op = g.create_op(*args, **kwargs)
  if len(op.outputs) == 1:
    return op.outputs[0]
  else:
    return op.outputs
         */


        [Ignore("")]
        [TestMethod]
        public void TestBasicWithConversion()
        {
            var g = tf.Graph().as_default();
            // Note: _apply_op can be replaced by g.create_op
            var a = g.create_op("FloatOutput", new Tensor[] { }, new[] { TF_DataType.TF_FLOAT });
            // TODO: ConvertibleObj, see original source below
            /*
    def testBasicWithConversion(self):
    g = ops.Graph()
    a = _apply_op(g, "FloatOutput", [], [dtypes.float32])

    class ConvertibleObj(object):

      def _as_graph_element(self):
        return a

    with g.control_dependencies([ConvertibleObj()]):
      c = _apply_op(g, "FloatOutput", [], [dtypes.float32])

    self.assertEqual(c.op.control_inputs, [a.op])
             */
        }
        
        //[Ignore]
        [TestMethod()]
        public void TestNested()
        {
            var g = ops.get_default_graph();
            var a_1 = constant_op.constant(1.0);
            var a_2 = constant_op.constant(3.0);
            var a_3 = constant_op.constant(4.0);
            var a_4 = constant_op.constant(5.0);
            Operation b_1 = null, b_2 = null;
            with(g.control_dependencies(new ITensorOrOperation[] { a_1, a_2, a_3, a_4 }), ctrl =>
            {
                b_1 = constant_op.constant(6.0);
            });
            with(g.control_dependencies(new ITensorOrOperation[] { a_1 }), ctrl1 =>
            {
                with(g.control_dependencies(new ITensorOrOperation[] { a_2 }), ctrl2 =>
                {
                    with(g.control_dependencies(new ITensorOrOperation[] { a_3 }), ctrl3 =>
                    {
                        with(g.control_dependencies(new ITensorOrOperation[] { a_4 }), ctrl4 =>
                        {
                            b_2 = constant_op.constant(7.0);
                        });
                    });
                });
            });
            AssertItemsEqual(new[] {a_1.op, a_2.op, a_3.op, a_4.op}, b_1.op.control_inputs);
            AssertItemsEqual(b_1.op.control_inputs, b_2.op.control_inputs);
        }


        [Ignore("will fail due to unsupported op 'FloatOutput'")]
        [TestMethod]
        public void TestClear()
        {
            /*
  def testClear(self):
    g = ops.Graph()
    a_1 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
    a_2 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
    a_3 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
    a_4 = _apply_op(g, "FloatOutput", [], [dtypes.float32])

    with g.control_dependencies([a_1]):
      with g.control_dependencies([a_2]):
        with g.control_dependencies(None):
          with g.control_dependencies([a_3]):
            with g.control_dependencies([a_4]):
              # deps [a_3, a_4]
              b_3_4 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
            # deps = [a_3]
            b_3 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
          # deps back to None
          b_none = _apply_op(g, "FloatOutput", [], [dtypes.float32])
        # deps back to [a_1, a_2]
        b_1_2 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
      # deps back to [a_1]
      b_1 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
      with g.control_dependencies(None):
        # deps are None again
        b_none2 = _apply_op(g, "FloatOutput", [], [dtypes.float32])

    self.assertItemsEqual([a_3.op, a_4.op], b_3_4.op.control_inputs)
    self.assertItemsEqual([a_3.op], b_3.op.control_inputs)
    self.assertItemsEqual([], b_none.op.control_inputs)
    self.assertItemsEqual([a_1.op, a_2.op], b_1_2.op.control_inputs)
    self.assertItemsEqual([a_1.op], b_1.op.control_inputs)
    self.assertItemsEqual([], b_none2.op.control_inputs)
             */
        }

        [Ignore("will fail due to unsupported op 'FloatOutput'")]
        [TestMethod]
        public void TestComplex()
        {
            /*
  def testComplex(self):
    g = ops.Graph()

    # Usage pattern:
    # * Nodes a_i are constants defined at the outermost scope, and are used
    #   as control inputs for the ith nested scope.
    # * Nodes b_i are defined as Mul(a_3, a_4) at each scope.
    # * Nodes c_i are defined as Mul(a_1, b_1) at each scope.
    # * Nodes d_i are defined as Mul(b_i, c_i) at each scope.
    # * Nodes e_i are defined as Mul(e_i-1, e_i-1) at each scope i > 1.

    a_1 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
    a_2 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
    a_3 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
    a_4 = _apply_op(g, "FloatOutput", [], [dtypes.float32])

    with g.control_dependencies([a_1]):
      b_1 = _apply_op(g, "TwoFloatInputsFloatOutput", [a_3, a_4],
                      [dtypes.float32])
      c_1 = _apply_op(g, "TwoFloatInputsFloatOutput", [a_1, b_1],
                      [dtypes.float32])
      d_1 = _apply_op(g, "TwoFloatInputsFloatOutput", [b_1, c_1],
                      [dtypes.float32])
      e_1 = _apply_op(g, "FloatOutput", [], [dtypes.float32])
      with g.control_dependencies([a_2]):
        b_2 = _apply_op(g, "TwoFloatInputsFloatOutput", [a_3, a_4],
                        [dtypes.float32])
        c_2 = _apply_op(g, "TwoFloatInputsFloatOutput", [a_1, b_1],
                        [dtypes.float32])
        d_2 = _apply_op(g, "TwoFloatInputsFloatOutput", [b_2, c_2],
                        [dtypes.float32])
        e_2 = _apply_op(g, "TwoFloatInputsFloatOutput", [e_1, e_1],
                        [dtypes.float32])
        with g.control_dependencies([a_3]):
          b_3 = _apply_op(g, "TwoFloatInputsFloatOutput", [a_3, a_4],
                          [dtypes.float32])
          c_3 = _apply_op(g, "TwoFloatInputsFloatOutput", [a_1, b_1],
                          [dtypes.float32])
          d_3 = _apply_op(g, "TwoFloatInputsFloatOutput", [b_3, c_3],
                          [dtypes.float32])
          e_3 = _apply_op(g, "TwoFloatInputsFloatOutput", [e_2, e_2],
                          [dtypes.float32])
          with g.control_dependencies([a_4]):
            b_4 = _apply_op(g, "TwoFloatInputsFloatOutput", [a_3, a_4],
                            [dtypes.float32])
            c_4 = _apply_op(g, "TwoFloatInputsFloatOutput", [a_1, b_1],
                            [dtypes.float32])
            d_4 = _apply_op(g, "TwoFloatInputsFloatOutput", [b_4, c_4],
                            [dtypes.float32])
            e_4 = _apply_op(g, "TwoFloatInputsFloatOutput", [e_3, e_3],
                            [dtypes.float32])

    self.assertItemsEqual([a_1.op], b_1.op.control_inputs)
    self.assertItemsEqual([a_1.op, a_2.op], b_2.op.control_inputs)
    self.assertItemsEqual([a_1.op, a_2.op], b_3.op.control_inputs)
    self.assertItemsEqual([a_1.op, a_2.op], b_4.op.control_inputs)

    self.assertItemsEqual([], c_1.op.control_inputs)
    self.assertItemsEqual([a_2.op], c_2.op.control_inputs)
    self.assertItemsEqual([a_2.op, a_3.op], c_3.op.control_inputs)
    self.assertItemsEqual([a_2.op, a_3.op, a_4.op], c_4.op.control_inputs)

    self.assertItemsEqual([], d_1.op.control_inputs)
    self.assertItemsEqual([], d_2.op.control_inputs)
    self.assertItemsEqual([], d_3.op.control_inputs)
    self.assertItemsEqual([], d_4.op.control_inputs)

    self.assertItemsEqual([a_1.op], e_1.op.control_inputs)
    self.assertItemsEqual([a_2.op], e_2.op.control_inputs)
    self.assertItemsEqual([a_3.op], e_3.op.control_inputs)
    self.assertItemsEqual([a_4.op], e_4.op.control_inputs)
             */
        }

        [Ignore("will fail due to unsupported op 'FloatOutput'")]
        [TestMethod]
        public void TestRepeatedDependency()
        {
            /*
  def testRepeatedDependency(self):
    g = ops.Graph()
    a = g.create_op("TwoFloatOutputs", [], [dtypes.float32, dtypes.float32])
    a_0, a_1 = a.outputs
    with g.control_dependencies([a_0]):
      b = _apply_op(g, "FloatOutput", [], [dtypes.float32])
      with g.control_dependencies([a_1]):
        c = _apply_op(g, "FloatOutput", [], [dtypes.float32])

    self.assertEqual(b.op.control_inputs, [a])
    self.assertEqual(c.op.control_inputs, [a])

  def testNoControlDependencyWithDataDependency(self):
    g = ops.Graph()
    a = _apply_op(g, "FloatOutput", [], [dtypes.float32])
    with g.control_dependencies([a]):
      b = _apply_op(g, "Identity", [a], [dtypes.float32])

    self.assertEqual(b.op.control_inputs, [])
             */
        }

    }
}
