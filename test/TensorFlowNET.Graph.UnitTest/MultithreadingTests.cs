using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow.NumPy;
using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest
{
    [TestClass]
    public class MultithreadingTests : GraphModeTestBase
    {
        [TestMethod]
        public void SessionCreation()
        {
            ops.uid(); //increment id by one

            MultiThreadedUnitTestExecuter.Run(8, Core);

            //the core method
            void Core(int tid)
            {
                Assert.IsNull(tf.peak_default_graph());

                var sess = tf.Session();
                var default_graph = tf.get_default_graph();
                var sess_graph = sess.graph;
                Assert.IsNotNull(default_graph);
                Assert.IsNotNull(sess_graph);
                Assert.AreEqual(default_graph, sess_graph);
            }
        }

        [TestMethod]
        public void SessionCreation_x2()
        {
            ops.uid(); //increment id by one

            MultiThreadedUnitTestExecuter.Run(16, Core);

            //the core method
            void Core(int tid)
            {
                Assert.IsNull(tf.peak_default_graph());
                //tf.Session created an other graph
                var sess = tf.Session();
                var default_graph = tf.get_default_graph();
                var sess_graph = sess.graph;
                Assert.IsNotNull(default_graph);
                Assert.IsNotNull(sess_graph);
                Assert.AreEqual(default_graph, sess_graph);
            }
        }

        [TestMethod]
        public void GraphCreation()
        {
            ops.uid(); //increment id by one

            MultiThreadedUnitTestExecuter.Run(8, Core);

            //the core method
            void Core(int tid)
            {
                Assert.IsNull(tf.peak_default_graph());
                var beforehand = tf.get_default_graph(); //this should create default automatically.
                beforehand.as_default();
                Assert.IsNotNull(tf.peak_default_graph());

                var sess = tf.Session();
                var default_graph = tf.peak_default_graph();
                var sess_graph = sess.graph;
                Assert.IsNotNull(default_graph);
                Assert.IsNotNull(sess_graph);
                Assert.AreEqual(default_graph, sess_graph);
            }
        }


        [TestMethod]
        public void Marshal_AllocHGlobal()
        {
            MultiThreadedUnitTestExecuter.Run(8, Core);

            //the core method
            void Core(int tid)
            {
                for (int i = 0; i < 100; i++)
                {
                    Marshal.FreeHGlobal(Marshal.AllocHGlobal(sizeof(int)));
                }
            }
        }

        [TestMethod]
        public void TensorCreation()
        {
            MultiThreadedUnitTestExecuter.Run(8, Core);

            //the core method
            void Core(int tid)
            {
                var sess = tf.Session();
                for (int i = 0; i < 100; i++)
                {
                    var t = new Tensor(1);
                }
            }
        }

        [TestMethod]
        public void TensorCreation_Array()
        {
            MultiThreadedUnitTestExecuter.Run(8, Core);

            //the core method
            void Core(int tid)
            {
                //tf.Session created an other graph
                var sess = tf.Session();
                for (int i = 0; i < 100; i++)
                {
                    var t = new Tensor(new int[] { 1, 2, 3 });
                }
            }
        }

        [TestMethod]
        public void SessionRun()
        {
            MultiThreadedUnitTestExecuter.Run(2, Core);

            //the core method
            void Core(int tid)
            {
                tf.compat.v1.disable_eager_execution();
                var graph = tf.Graph().as_default();

                //graph is created automatically to perform create these operations
                var a1 = tf.constant(new[] { 2f }, shape: new[] { 1 });
                var a2 = tf.constant(new[] { 3f }, shape: new[] { 1 });
                var math = a1 + a2;
                var sess = tf.Session(graph);
                for (int i = 0; i < 100; i++)
                {
                    var result = sess.run(math);
                    Assert.AreEqual(result[0], 5f);
                }
            }
        }

        [TestMethod]
        public void SessionRun_InsideSession()
        {
            MultiThreadedUnitTestExecuter.Run(8, Core);

            //the core method
            void Core(int tid)
            {
                tf.compat.v1.disable_eager_execution();
                var graph = tf.Graph().as_default();

                var sess = tf.Session(graph);
                Assert.IsNotNull(tf.get_default_graph());
                //graph is created automatically to perform create these operations
                var a1 = tf.constant(new[] { 2f }, shape: new[] { 1 });
                var a2 = tf.constant(new[] { 3f }, shape: new[] { 1 });
                var math = a1 + a2;

                var result = sess.run(math);
                Assert.AreEqual(result[0], 5f);
            }
        }

        [TestMethod]
        public void SessionRun_Initialization()
        {
            MultiThreadedUnitTestExecuter.Run(8, Core);

            //the core method
            void Core(int tid)
            {
                var sess = tf.Session();
                Assert.IsNotNull(tf.get_default_graph());
                //graph is created automatically to perform create these operations
                var a1 = tf.constant(new[] { 2f }, shape: new[] { 1 });
                var a2 = tf.constant(new[] { 3f }, shape: new[] { 1 });
                var math = a1 + a2;
            }
        }

        [TestMethod]
        public void SessionRun_Initialization_OutsideSession()
        {
            MultiThreadedUnitTestExecuter.Run(8, Core);

            //the core method
            void Core(int tid)
            {
                Assert.IsNull(tf.peak_default_graph());
                //graph is created automatically to perform create these operations
                var a1 = tf.constant(new[] { 2f }, shape: new[] { 1 });
                var a2 = tf.constant(new[] { 3f }, shape: new[] { 1 });
                var math = a1 + a2;
            }
        }

        [TestMethod]
        public void TF_GraphOperationByName()
        {
            MultiThreadedUnitTestExecuter.Run(8, Core);

            //the core method
            void Core(int tid)
            {
                Assert.IsNull(tf.peak_default_graph());

                tf.compat.v1.disable_eager_execution();
                var graph = tf.Graph().as_default();

                //graph is created automatically to perform create these operations
                var a1 = tf.constant(new[] { 2f }, shape: new[] { 1 });
                var a2 = tf.constant(new[] { 3f }, shape: new[] { 1 }, name: "ConstantK");
                var math = a1 + a2;
                for (int i = 0; i < 100; i++)
                {
                    var op = tf.get_default_graph().OperationByName("ConstantK");
                }
            }
        }

        private static readonly string modelPath = Path.GetFullPath("./Utilities/models/example1/");

        [Ignore]
        public void TF_GraphOperationByName_FromModel()
        {
            MultiThreadedUnitTestExecuter.Run(8, Core);

            //the core method
            void Core(int tid)
            {
                Console.WriteLine();
                for (int j = 0; j < 100; j++)
                {
                    var sess = Session.LoadFromSavedModel(modelPath).as_default();
                    var inputs = new[] { "sp", "fuel" };

                    var inp = inputs.Select(name => sess.graph.OperationByName(name).output).ToArray();
                    var outp = sess.graph.OperationByName("softmax_tensor").output;

                    for (var i = 0; i < 8; i++)
                    {
                        var data = new float[96];
                        FeedItem[] feeds = new FeedItem[2];

                        for (int f = 0; f < 2; f++)
                            feeds[f] = new FeedItem(inp[f], new NDArray(data));

                        sess.run(outp, feeds);
                    }
                }
            }
        }
    }
}