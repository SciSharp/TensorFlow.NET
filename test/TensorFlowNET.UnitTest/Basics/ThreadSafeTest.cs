using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.Basics
{
    [TestClass]
    public class ThreadSafeTest
    {
        [TestMethod]
        public void GraphWithMultiThreads()
        {
            List<Thread> threads = new List<Thread>();

            const int THREADS_COUNT = 5;

            for (int t = 0; t < THREADS_COUNT; t++)
            {
                Thread thread = new Thread(() =>
                {
                    Graph g = new Graph();
                    Session session = new Session(g);
                    session.as_default();
                    var input = tf.placeholder(tf.int32, shape: new Shape(6));
                    var op = tf.reshape(input, new int[] { 2, 3 });
                });
                thread.Start();
                threads.Add(thread);
            }

            threads.ForEach(t => t.Join());
        }
    }
}
