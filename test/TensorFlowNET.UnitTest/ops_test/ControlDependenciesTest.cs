using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow;
using Tensorflow.Eager;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.ops_test
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
            var g = tf.Graph().as_default();
            Tensor a = null, b = null, c = null, d = null, e = null;

            a = constant_op.constant(1.0);
            b = constant_op.constant(1.0);
            tf_with(g.control_dependencies(new[] { a }), x =>
            {
                c = constant_op.constant(1.0);
                d = array_ops.identity(b);
                e = array_ops.identity(c);
            });

            Assert.IsTrue(Enumerable.SequenceEqual(c.op.control_inputs, new[] { a.op }));
            Assert.IsTrue(Enumerable.SequenceEqual(d.op.control_inputs, new[] { a.op }));
            // e should be dominated by c.
            Assert.AreEqual(0, e.op.control_inputs.Length);
        }

        [Ignore("Future is not supported yet")]
        [TestMethod]
        public void TestEager()
        {
            Tensor a = null, c = null;
            object b = null;
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
                    a = constant_op.constant(1.0);
                    b = future; // <--- {henon} obviously, this doesn't compile, looks like control_dependencies needs to be able to take callables as well. 
                    tf_with(ops.control_dependencies(new object[] { a, b }), ctrl =>
                      {
                          return c = constant_op.constant(3.0);
                      });
                    Assert.AreEqual(calls, 1);
                }
                else
                {
                    var g = tf.Graph().as_default();
                    a = constant_op.constant(1.0);
                    var b1 = future();
                    tf_with(g.control_dependencies(new[] { a, b }), ctrl =>
                    {
                        c = constant_op.constant(3.0);
                    });
                    Assert.IsTrue(Enumerable.SequenceEqual(c.op.control_inputs, new[] { a.op, b1.op }));
                    Assert.AreEqual(1, calls);
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


        [Ignore("How to port the ConvertibleObj?")]
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

        [TestMethod]
        public void TestNested()
        {
            var g = tf.Graph().as_default();
            var a_1 = constant_op.constant(1.0);
            var a_2 = constant_op.constant(3.0);
            var a_3 = constant_op.constant(4.0);
            var a_4 = constant_op.constant(5.0);
            Tensor b_1 = null, b_2 = null;
            tf_with(g.control_dependencies(new[] { a_1, a_2, a_3, a_4 }), ctrl =>
             {
                 b_1 = constant_op.constant(6.0);
             });
            tf_with(g.control_dependencies(new[] { a_1 }), ctrl1 =>
             {
                 tf_with(g.control_dependencies(new[] { a_2 }), ctrl2 =>
                 {
                     tf_with(g.control_dependencies(new[] { a_3 }), ctrl3 =>
                     {
                         tf_with(g.control_dependencies(new[] { a_4 }), ctrl4 =>
                         {
                             b_2 = constant_op.constant(7.0);
                         });
                     });
                 });
             });
            //var z=tf.add(a_1, tf.multiply(b_2, b_1));
            //with(g.control_dependencies(new[] {z}), ctrl =>
            //{
            //    var z1 = tf.add(a_3, tf.multiply(a_4, a_2));
            //});
            //tf.train.export_meta_graph(@"D:\dev\tensorboard\logdir\sharp.meta", as_text: false);
            assertItemsEqual(b_1.op.control_inputs, new[] { a_1.op, a_2.op, a_3.op, a_4.op });
            assertItemsEqual(b_2.op.control_inputs, b_1.op.control_inputs);
        }

        [TestMethod]
        public void TestClear()
        {
            var g = tf.Graph().as_default();
            var a_1 = constant_op.constant(1.0);
            var a_2 = constant_op.constant(3.0);
            var a_3 = constant_op.constant(4.0);
            var a_4 = constant_op.constant(5.0);
            Operation b_3_4 = null, b_3 = null, b_none = null, b_1 = null, b_1_2 = null, b_none2 = null;
            tf_with(g.control_dependencies(new[] { a_1 }), ctrl1 =>
            {
                tf_with(g.control_dependencies(new[] { a_2 }), ctrl2 =>
                {
                    tf_with(g.control_dependencies(null), ctrl3 =>
                    {
                        tf_with(g.control_dependencies(new[] { a_3 }), ctrl4 =>
                        {
                            tf_with(g.control_dependencies(new[] { a_4 }), ctrl5 =>
                            {
                                // deps [a_3, a_4]
                                b_3_4 = constant_op.constant(7.0);
                            });
                            // deps = [a_3]
                            b_3 = constant_op.constant(8.0);
                        });
                        // deps back to None
                        b_none = constant_op.constant(9.0);
                    });
                    // deps back to [a_1, a_2]
                    b_1_2 = constant_op.constant(10.0);
                });
                // deps back to [a_1]
                b_1 = constant_op.constant(11.0);
                tf_with(g.control_dependencies(null), ctrl6 =>
                {
                    // deps are None again
                    b_none2 = constant_op.constant(12.0);
                });
            });
            // Note assertItemsEqual(given, expected), expected and given parameters should be swapped below 
            assertItemsEqual(new[] { a_3.op, a_4.op }, b_3_4.op.control_inputs);
            assertItemsEqual(new[] { a_3.op }, b_3.op.control_inputs);
            assertItemsEqual(new object[0], b_none.op.control_inputs);
            assertItemsEqual(new[] { a_1.op, a_2.op }, b_1_2.op.control_inputs);
            assertItemsEqual(new[] { a_1.op }, b_1.op.control_inputs);
            assertItemsEqual(new object[0], b_none2.op.control_inputs);
        }

        [TestMethod]
        public void TestComplex()
        {
            var g = tf.Graph().as_default();
            // Usage pattern:
            // * Nodes a_i are constants defined at the outermost scope, and are used
            // as control inputs for the ith nested scope.
            // * Nodes b_i are defined as Mul(a_3, a_4) at each scope.
            // * Nodes c_i are defined as Mul(a_1, b_1) at each scope.
            // * Nodes d_i are defined as Mul(b_i, c_i) at each scope.
            // * Nodes e_i are defined as Mul(e_i-1, e_i-1) at each scope i > 1.
            var a_1 = constant_op.constant(1.0);
            var a_2 = constant_op.constant(2.0);
            var a_3 = constant_op.constant(3.0);
            var a_4 = constant_op.constant(4.0);
            Operation b_1 = null, b_2 = null, b_3 = null, b_4 = null;
            Operation c_1 = null, c_2 = null, c_3 = null, c_4 = null;
            Operation d_1 = null, d_2 = null, d_3 = null, d_4 = null;
            Operation e_1 = null, e_2 = null, e_3 = null, e_4 = null;
            tf_with(g.control_dependencies(new[] { a_1 }), ctrl1 =>
            {
                b_1 = tf.multiply(a_3, a_4);
                c_1 = tf.multiply(a_1, b_1.output);
                d_1 = tf.multiply(b_1.output, c_1.output);
                e_1 = constant_op.constant(5.0);
                tf_with(g.control_dependencies(new[] { a_2 }), ctrl2 =>
                {
                    b_2 = tf.multiply(a_3, a_4);
                    c_2 = tf.multiply(a_1, b_1.output);
                    d_2 = tf.multiply(b_2.output, c_2.output);
                    e_2 = tf.multiply(e_1.output, e_1.output);
                    tf_with(g.control_dependencies(new[] { a_3 }), ctrl3 =>
                    {
                        b_3 = tf.multiply(a_3, a_4);
                        c_3 = tf.multiply(a_1, b_1.output);
                        d_3 = tf.multiply(b_3.output, c_3.output);
                        e_3 = tf.multiply(e_2.output, e_2.output);
                        tf_with(g.control_dependencies(new[] { a_4 }), ctrl4 =>
                        {
                            b_4 = tf.multiply(a_3, a_4);
                            c_4 = tf.multiply(a_1, b_1.output);
                            d_4 = tf.multiply(b_4.output, c_4.output);
                            e_4 = tf.multiply(e_3.output, e_3.output);
                        });
                    });
                });
            });

            // Note assertItemsEqual(given, expected), expected and given parameters should be swapped below 
            assertItemsEqual(new[] {a_1.op}, b_1.op.control_inputs);
            assertItemsEqual(new[] {a_1.op, a_2.op}, b_2.op.control_inputs);
            assertItemsEqual(new[] { a_1.op, a_2.op}, b_3.op.control_inputs);
            assertItemsEqual(new[] {a_1.op, a_2.op}, b_4.op.control_inputs);

            assertItemsEqual(new object[0], c_1.op.control_inputs);
            assertItemsEqual(new[] {a_2.op}, c_2.op.control_inputs);
            assertItemsEqual(new[] {a_2.op, a_3.op}, c_3.op.control_inputs);
            assertItemsEqual(new[] {a_2.op, a_3.op, a_4.op}, c_4.op.control_inputs);

            assertItemsEqual(new object[0], d_1.op.control_inputs);
            assertItemsEqual(new object[0], d_2.op.control_inputs);
            assertItemsEqual(new object[0], d_3.op.control_inputs);
            assertItemsEqual(new object[0], d_4.op.control_inputs);

            assertItemsEqual(new[] {a_1.op}, e_1.op.control_inputs);
            assertItemsEqual(new[] {a_2.op}, e_2.op.control_inputs);
            assertItemsEqual(new[] {a_3.op}, e_3.op.control_inputs);
            assertItemsEqual(new[] {a_4.op}, e_4.op.control_inputs);
        }

        [Ignore("Don't know how to create an operation with two outputs")]
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
 
             */
        }

        [TestMethod]
        public void TestNoControlDependencyWithDataDependency()
        {
            var g = tf.Graph().as_default();
            Operation b = null;
            var a = constant_op.constant(100.0);
            tf_with(g.control_dependencies(new[] { a }), ctrl1 =>
            {
                b = array_ops.identity(a);
            });
            Assert.AreEqual(0, b.op.control_inputs.Length);
        }

    }
}
