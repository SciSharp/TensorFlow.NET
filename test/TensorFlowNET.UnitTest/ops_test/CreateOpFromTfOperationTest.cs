using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow;
using Tensorflow.Operations;
using Tensorflow.UnitTest;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.ops_test
{
    /// <summary>
    /// excerpt of tensorflow/python/framework/ops_test.py
    ///         # These cases test the private Graph._create_op_from_tf_operation
    ///         # method. Arguably we should only test the public APIs that depend on this
    ///         # method. However, this logic is complex and tricky, and it can be difficult to
    ///         # ascertain if we have adequate coverage (e.g. a graph may run successfully if
    ///         # the control flow context isn't set properly, but a more complicated use case
    ///         # that might not be obvious to test will fail). Thus we instead explicitly test
    ///         # the low-level behavior.
    /// </summary>
    [Ignore]
    [TestClass]
    public class CreateOpFromTfOperationTest : GraphModeTestBase
    {

        [TestMethod]
        public void TestShape()
        {
            using (var g = tf.Graph().as_default())
            {
                var x = constant_op.constant(new[,] {{1, 2, 3}, {4, 5, 6}});
                var (c_op, op_desc) = ops._create_c_op(g, ops._NodeDef("Identity", "myop"), new[] {x}, new Operation[0]);
                var op = g._create_op_from_tf_operation(c_op);

                Assert.AreEqual("myop", op.name);
                Assert.AreEqual("Identity", op.type);
                Assert.AreEqual(1, len(op.outputs));
                assertItemsEqual(new[] {2, 3}, op.outputs[0].shape);
            }
        }

        [TestMethod]
        public void TestUniqueName()
        {
            var graph = tf.Graph().as_default();
            //var (c_op,op_desc) = ops._create_c_op(g, ops._NodeDef("Const", "myop"), new Tensor[0], new Operation[0]);
            //var (c_op2, op_desc1) = ops._create_c_op(g, ops._NodeDef("Const", "myop_1"), new Tensor[0], new Operation[0]);
            //var op = g._create_op_from_tf_operation(c_op);
            //var op2 = g._create_op_from_tf_operation(c_op2);
            var op = constant_op.constant(0, name: "myop").op;
            var op2 = constant_op.constant(0, name: "myop_1").op;

            // Create ops with same names as op1 and op2. We expect the new names to be
            // uniquified.
            var op3 = constant_op.constant(0, name: "myop").op;
            var op4 = constant_op.constant(0, name: "myop_1").op;

            self.assertEqual(op.name, "myop");
            self.assertEqual(op2.name, "myop_1");
            self.assertEqual(op3.name, "myop_2");
            self.assertEqual(op4.name, "myop_1_1");
        }

        [Ignore("need tesnroflow expose UpdateEdge API")]
        [TestMethod]
        public void TestCond()
        {
            var g = tf.Graph().as_default();
            var x = constant_op.constant(10);

            var true_fn = new Func<Tensor>(() =>
            {
                var c_op = ops._create_c_op(g, ops._NodeDef("Identity", "cond/myop"), new[] { x }, new Operation[0]);
                var new_ops = g._add_new_tf_operations();
                self.assertEqual(len(new_ops), 1);
                return x;
            });

            control_flow_ops.cond(x < 10, true_fn, () => x);

            var op = g.get_operation_by_name("cond/myop");

            //tf.train.export_meta_graph(@"D:\dev\tensorboard\logdir\sharp.meta.txt", as_text:true);
            //tf.train.export_meta_graph(@"D:\dev\tensorboard\logdir\sharp.meta", as_text: false);

            self.assertIsNotNone(op);
            self.assertEqual(op.name, "cond/myop");
            self.assertEqual(op.type, "Identity");
            //self.assertEqual(op.outputs, new object[0]);
            var op_input = op.inputs[0].op;
            self.assertEqual(op_input.type, "Switch");
            self.assertEqual(op_input.inputs[0].name, x.name);
            self.assertEqual(op.graph, g);
            self.assertIsNotNone(op._get_control_flow_context());
            var cond_text = op._get_control_flow_context() as ControlFlowContext;
            self.assertEqual(cond_text.Name, "cond/cond_text");
        }

        [Ignore("Todo: Port")]
        [TestMethod]
        public void TestWhileLoop()
        {
            var graph = tf.Graph().as_default();
            Operation x=null;
            x = constant_op.constant(42);
            var body = new Func<int, int>(i =>
            {
                ops._create_c_op(ops.get_default_graph(), ops._NodeDef("Identity", "myloop/myop"), new[] { x.output },
                    new Operation[0]);
                var new_ops = graph._add_new_tf_operations();
                self.assertEqual(len(new_ops), 1);
                return i;
            });
            // TODO: port control_flow_ops.while_loop
            //control_flow_ops.while_loop( i => i < 10, body, new int[]{0}, name = "myloop");
            var op = graph.get_operation_by_name("myloop/myop");
            self.assertIsNotNone(op);
            self.assertEqual(op.name, "myloop/myop");
            self.assertEqual(op.type, "Identity");
            self.assertEqual(op.outputs.Length, 0);
            var op_input = op.inputs[0].op;
            self.assertEqual(op_input.type, "Enter");
            self.assertItemsEqual(op_input.inputs.OfType<Operation>().ToArray(), new[] {x});
            self.assertEqual(op.graph, graph);
            self.assertIsNotNone(op._get_control_flow_context());
            self.assertEqual(((ControlFlowContext)op._get_control_flow_context()).Name, "myloop/while_context");
            /*
                  @test_util.run_v1_only("b/120545219")
                  def testWhileLoop(self):
                    g = ops.Graph()
                    with g.as_default():
                      x = test_ops.int_output()

                      def body(i):
                        ops._create_c_op(ops.get_default_graph(),
                                         ops._NodeDef("IntInput", "myloop/myop"), [x], [])
                        new_ops = g._add_new_tf_operations()
                        self.assertEqual(len(new_ops), 1)
                        return i

                      control_flow_ops.while_loop(lambda i: i < 10, body, [0], name="myloop")

                    op = g.get_operation_by_name("myloop/myop")
                    self.assertIsNotNone(op)
                    self.assertEqual(op.name, "myloop/myop")
                    self.assertEqual(op.type, "IntInput")
                    self.assertEqual(op.outputs, [])
                    op_input = op.inputs[0].op
                    self.assertEqual(op_input.type, "Enter")
                    self.assertEqual(list(op_input.inputs), [x])
                    self.assertEqual(op.graph, g)
                    # pylint: disable=protected-access
                    self.assertIsNotNone(op._get_control_flow_context())
                    self.assertEqual(op._get_control_flow_context().name,
                                     "myloop/while_context")
                    # pylint: enable=protected-access
                    */
        }

        [Ignore("Todo: Port")]
        [TestMethod]
        public void TestWhileLoopWithInternalControlDep()
        {
            /*
@test_util.run_v1_only("b/120545219")
                  def testWhileLoopWithInternalControlDep(self):
                    g = ops.Graph()
                    with g.as_default():
                      x = test_ops.int_output()

                      def body(i):
                        c = constant_op.constant(1.0, name="c")
                        ops._create_c_op(ops.get_default_graph(),
                                         ops._NodeDef("IntInput", "myloop/myop"), [x], [])
                        with ops.control_dependencies([c]):
                          new_ops = g._add_new_tf_operations()
                          self.assertEqual(len(new_ops), 1)
                        return i

                      control_flow_ops.while_loop(lambda i: i < 10, body, [0], name="myloop")

                    op = g.get_operation_by_name("myloop/myop")
                    self.assertIsNotNone(op)
                    c = g.get_operation_by_name("myloop/c")
                    self.assertIsNotNone(c)
                    # Internal control dep is preserved
                    self.assertEqual(op.control_inputs, [c])
                    */
        }

        [Ignore("Todo: Port")]
        [TestMethod]
        public void TestWhileLoopWithExternalControlDep()
        {
            /*
                  @test_util.run_v1_only("b/120545219")
                  def testWhileLoopWithExternalControlDep(self):
                    g = ops.Graph()
                    with g.as_default():
                      x = test_ops.int_output()
                      c = constant_op.constant(1.0)

                      def body(i):
                        ops._create_c_op(ops.get_default_graph(),
                                         ops._NodeDef("IntInput", "myloop/myop"), [x], [])
                        with ops.control_dependencies([c]):
                          new_ops = g._add_new_tf_operations()
                          self.assertEqual(len(new_ops), 1)
                        return i

                      control_flow_ops.while_loop(lambda i: i < 10, body, [0], name="myloop")

                    op = g.get_operation_by_name("myloop/myop")
                    self.assertIsNotNone(op)
                    # External control dep is removed and replaced with internal control dep
                    self.assertNotEqual(op.control_inputs[0], c.op)
                    self.assertIsNotNone(op.control_inputs[0]._get_control_flow_context())
                    */
        }

    }
}
