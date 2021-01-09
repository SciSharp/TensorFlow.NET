using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Tensorflow;
using Tensorflow.UnitTest;
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
                tf.peak_default_graph().Should().BeNull();

                using (var sess = tf.Session())
                {
                    var default_graph = tf.peak_default_graph();
                    var sess_graph = sess.GetPrivate<Graph>("_graph");
                    sess_graph.Should().NotBeNull();
                    default_graph.Should().NotBeNull()
                        .And.BeEquivalentTo(sess_graph);
                }
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
                tf.peak_default_graph().Should().BeNull();
                //tf.Session created an other graph
                using (var sess = tf.Session())
                {
                    var default_graph = tf.peak_default_graph();
                    var sess_graph = sess.GetPrivate<Graph>("_graph");
                    sess_graph.Should().NotBeNull();
                    default_graph.Should().NotBeNull()
                        .And.BeEquivalentTo(sess_graph);
                }
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
                tf.peak_default_graph().Should().BeNull();
                var beforehand = tf.get_default_graph(); //this should create default automatically.
                beforehand.graph_key.Should().NotContain("-0/", "Already created a graph in an other thread.");
                beforehand.as_default();
                tf.peak_default_graph().Should().NotBeNull();

                using (var sess = tf.Session())
                {
                    var default_graph = tf.peak_default_graph();
                    var sess_graph = sess.GetPrivate<Graph>("_graph");
                    sess_graph.Should().NotBeNull();
                    default_graph.Should().NotBeNull()
                        .And.BeEquivalentTo(sess_graph);

                    Console.WriteLine($"{tid}-{default_graph.graph_key}");

                    //var result = sess.run(new object[] {g, a});
                    //var actualDeriv = result[0].GetData<float>()[0];
                    //var actual = result[1].GetData<float>()[0];
                }
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
            //lock (Locks.ProcessWide)
            //    tf.Session(); //create one to increase next id to 1.

            MultiThreadedUnitTestExecuter.Run(8, Core);

            //the core method
            void Core(int tid)
            {
                using (var sess = tf.Session())
                {
                    Tensor t = null;
                    for (int i = 0; i < 100; i++)
                    {
                        t = new Tensor(1);
                    }
                }
            }
        }

        [TestMethod]
        public void TensorCreation_Array()
        {
            //lock (Locks.ProcessWide)
            //    tf.Session(); //create one to increase next id to 1.

            MultiThreadedUnitTestExecuter.Run(8, Core);

            //the core method
            void Core(int tid)
            {
                //tf.Session created an other graph
                using (var sess = tf.Session())
                {
                    for (int i = 0; i < 100; i++)
                    {
                        var t = new Tensor(new int[] { 1, 2, 3 });
                    }
                }
            }
        }

        [TestMethod]
        public void TensorCreation_Undressed()
        {
            //lock (Locks.ProcessWide)
            //    tf.Session(); //create one to increase next id to 1.

            MultiThreadedUnitTestExecuter.Run(8, Core);

            //the core method
            unsafe void Core(int tid)
            {
                using (var sess = tf.Session())
                {
                    for (int i = 0; i < 100; i++)
                    {
                        var v = (int*)Marshal.AllocHGlobal(sizeof(int));
                        c_api.DeallocatorArgs _deallocatorArgs = new c_api.DeallocatorArgs();
                        var handle = c_api.TF_NewTensor(typeof(int).as_dtype(), dims: new long[0], num_dims: 0,
                            data: (IntPtr)v, len: (UIntPtr)sizeof(int),
                            deallocator: (IntPtr data, IntPtr size, ref c_api.DeallocatorArgs args) => Marshal.FreeHGlobal(data),
                            ref _deallocatorArgs);
                        c_api.TF_DeleteTensor(handle);
                    }
                }
            }
        }

        [TestMethod]
        public void SessionRun()
        {
            MultiThreadedUnitTestExecuter.Run(8, Core);

            //the core method
            void Core(int tid)
            {
                tf.peak_default_graph().Should().BeNull();
                //graph is created automatically to perform create these operations
                var a1 = tf.constant(new[] { 2f }, shape: new[] { 1 });
                var a2 = tf.constant(new[] { 3f }, shape: new[] { 1 });
                var math = a1 + a2;
                for (int i = 0; i < 100; i++)
                {
                    using (var sess = tf.Session())
                    {
                        sess.run(math).GetAtIndex<float>(0).Should().Be(5);
                    }
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
                using (var sess = tf.Session())
                {
                    tf.peak_default_graph().Should().NotBeNull();
                    //graph is created automatically to perform create these operations
                    var a1 = tf.constant(new[] { 2f }, shape: new[] { 1 });
                    var a2 = tf.constant(new[] { 3f }, shape: new[] { 1 });
                    var math = a1 + a2;

                    var result = sess.run(math);
                    result[0].GetAtIndex<float>(0).Should().Be(5);
                }
            }
        }

        [TestMethod]
        public void SessionRun_Initialization()
        {
            MultiThreadedUnitTestExecuter.Run(8, Core);

            //the core method
            void Core(int tid)
            {
                using (var sess = tf.Session())
                {
                    tf.peak_default_graph().Should().NotBeNull();
                    //graph is created automatically to perform create these operations
                    var a1 = tf.constant(new[] { 2f }, shape: new[] { 1 });
                    var a2 = tf.constant(new[] { 3f }, shape: new[] { 1 });
                    var math = a1 + a2;
                }
            }
        }

        [TestMethod]
        public void SessionRun_Initialization_OutsideSession()
        {
            MultiThreadedUnitTestExecuter.Run(8, Core);

            //the core method
            void Core(int tid)
            {
                tf.peak_default_graph().Should().BeNull();
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
                tf.peak_default_graph().Should().BeNull();
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
        [TestMethod]
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