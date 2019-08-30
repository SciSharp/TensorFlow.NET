using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest
{
    [TestClass]
    public class MultithreadingTests
    {
        [TestMethod]
        public void SessionCreation()
        {
            tf.Session(); //create one to increase next id to 1.

            MultiThreadedUnitTestExecuter.Run(8, Core);

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
            tf.Graph(); //create one to increase next id to 1.

            MultiThreadedUnitTestExecuter.Run(8, Core);

            //the core method
            void Core(int tid)
            {
                tf.peak_default_graph().Should().BeNull();
                var beforehand = tf.get_default_graph(); //this should create default automatically.
                beforehand.graph_key.Should().NotContain("0", "Already created a graph in an other thread.");
                tf.peak_default_graph().Should().NotBeNull();

                using (var sess = tf.Session())
                {
                    var default_graph = tf.peak_default_graph();
                    var sess_graph = sess.GetPrivate<Graph>("_graph");
                    sess_graph.Should().NotBeNull();
                    default_graph.Should().NotBeNull()
                        .And.BeEquivalentTo(sess_graph)
                        .And.BeEquivalentTo(beforehand);

                    Console.WriteLine($"{tid}-{default_graph.graph_key}");

                    //var result = sess.run(new object[] {g, a});
                    //var actualDeriv = result[0].GetData<float>()[0];
                    //var actual = result[1].GetData<float>()[0];
                }
            }
        }
    }
}