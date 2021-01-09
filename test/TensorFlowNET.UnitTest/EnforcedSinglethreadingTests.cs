using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Diagnostics;
using System.Threading;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest
{
    [TestClass]
    public class EnforcedSinglethreadingTests
    {
        private static readonly object _singlethreadLocker = new object();

        /// <summary>Initializes a new instance of the <see cref="T:System.Object" /> class.</summary>
        public EnforcedSinglethreadingTests()
        {
            ops.IsSingleThreaded = true;
        }

        [TestMethod, Ignore("Has to be tested manually.")]
        public void SessionCreation()
        {
            lock (_singlethreadLocker)
            {
                ops.IsSingleThreaded.Should().BeTrue();

                ops.uid(); //increment id by one

                //the core method
                tf.peak_default_graph().Should().BeNull();

                using (var sess = tf.Session())
                {
                    var default_graph = tf.peak_default_graph();
                    var sess_graph = sess.GetPrivate<Graph>("_graph");
                    sess_graph.Should().NotBeNull();
                    default_graph.Should().NotBeNull()
                        .And.BeEquivalentTo(sess_graph);

                    var (graph, session) = Parallely(() => (tf.get_default_graph(), tf.get_default_session()));

                    graph.Should().BeEquivalentTo(default_graph);
                    session.Should().BeEquivalentTo(sess);
                }
            }
        }

        T Parallely<T>(Func<T> fnc)
        {
            var mrh = new ManualResetEventSlim();
            T ret = default;
            Exception e = default;
            new Thread(() =>
            {
                try
                {
                    ret = fnc();
                }
                catch (Exception ee)
                {
                    e = ee;
                    throw;
                }
                finally
                {
                    mrh.Set();
                }
            }).Start();

            if (!Debugger.IsAttached)
                mrh.Wait(10000).Should().BeTrue();
            else
                mrh.Wait(-1);
            e.Should().BeNull(e?.ToString());
            return ret;
        }

        void Parallely(Action fnc)
        {
            var mrh = new ManualResetEventSlim();
            Exception e = default;
            new Thread(() =>
            {
                try
                {
                    fnc();
                }
                catch (Exception ee)
                {
                    e = ee;
                    throw;
                }
                finally
                {
                    mrh.Set();
                }
            }).Start();

            mrh.Wait(10000).Should().BeTrue();
            e.Should().BeNull(e.ToString());
        }
    }
}