using System;
using System.Diagnostics;
using System.Threading;

namespace TensorFlowNET.UnitTest
{
    public delegate void MultiThreadedTestDelegate(int threadid);

    /// <summary>
    ///     Creates a synchronized eco-system of running code.
    /// </summary>
    public class MultiThreadedUnitTestExecuter : IDisposable
    {
        public int ThreadCount { get; }
        public Thread[] Threads { get; }
        public Exception[] Exceptions { get; }
        private readonly SemaphoreSlim barrier_threadstarted;
        private readonly ManualResetEventSlim barrier_corestart;
        private readonly SemaphoreSlim done_barrier2;

        public Action<MultiThreadedUnitTestExecuter> PostRun { get; set; }

        #region Static

        [DebuggerHidden]
        public static void Run(int threadCount, MultiThreadedTestDelegate workload)
        {
            if (workload == null) throw new ArgumentNullException(nameof(workload));
            if (threadCount <= 0) throw new ArgumentOutOfRangeException(nameof(threadCount));
            new MultiThreadedUnitTestExecuter(threadCount).Run(workload);
        }

        [DebuggerHidden]
        public static void Run(int threadCount, params MultiThreadedTestDelegate[] workloads)
        {
            if (workloads == null) throw new ArgumentNullException(nameof(workloads));
            if (workloads.Length == 0) throw new ArgumentException("Value cannot be an empty collection.", nameof(workloads));
            if (threadCount <= 0) throw new ArgumentOutOfRangeException(nameof(threadCount));
            new MultiThreadedUnitTestExecuter(threadCount).Run(workloads);
        }

        [DebuggerHidden]
        public static void Run(int threadCount, MultiThreadedTestDelegate workload, Action<MultiThreadedUnitTestExecuter> postRun)
        {
            if (workload == null) throw new ArgumentNullException(nameof(workload));
            if (postRun == null) throw new ArgumentNullException(nameof(postRun));
            if (threadCount <= 0) throw new ArgumentOutOfRangeException(nameof(threadCount));
            new MultiThreadedUnitTestExecuter(threadCount) { PostRun = postRun }.Run(workload);
        }

        #endregion


        /// <summary>Initializes a new instance of the <see cref="T:System.Object"></see> class.</summary>
        public MultiThreadedUnitTestExecuter(int threadCount)
        {
            if (threadCount <= 0)
                throw new ArgumentOutOfRangeException(nameof(threadCount));
            ThreadCount = threadCount;
            Threads = new Thread[ThreadCount];
            Exceptions = new Exception[ThreadCount];
            done_barrier2 = new SemaphoreSlim(0, threadCount);
            barrier_corestart = new ManualResetEventSlim();
            barrier_threadstarted = new SemaphoreSlim(0, threadCount);
        }

        [DebuggerHidden]
        public void Run(params MultiThreadedTestDelegate[] workloads)
        {
            if (workloads == null)
                throw new ArgumentNullException(nameof(workloads));
            if (workloads.Length != 1 && workloads.Length % ThreadCount != 0)
                throw new InvalidOperationException($"Run method must accept either 1 workload or n-threads workloads. Got {workloads.Length} workloads.");

            if (ThreadCount == 1)
            {
                Exception ex = null;
                new Thread(() =>
                {
                    try
                    {
                        workloads[0](0);
                    }
                    catch (Exception e)
                    {
                        if (Debugger.IsAttached)
                            throw;
                        ex = e;
                    }
                    finally
                    {
                        done_barrier2.Release(1);
                    }
                }).Start();

                done_barrier2.Wait();

                if (ex != null)
                    throw new Exception($"Thread 0 has failed: ", ex);

                PostRun?.Invoke(this);

                return;
            }

            //thread core
            Exception ThreadCore(MultiThreadedTestDelegate core, int threadid)
            {
                barrier_threadstarted.Release(1);
                barrier_corestart.Wait();
                //workload
                try
                {
                    core(threadid);
                }
                catch (Exception e)
                {
                    if (Debugger.IsAttached)
                        throw;
                    return e;
                }
                finally
                {
                    done_barrier2.Release(1);
                }

                return null;
            }

            //initialize all threads
            if (workloads.Length == 1)
            {
                var workload = workloads[0];
                for (int i = 0; i < ThreadCount; i++)
                {
                    var i_local = i;
                    Threads[i] = new Thread(() => Exceptions[i_local] = ThreadCore(workload, i_local));
                }
            }
            else
            {
                for (int i = 0; i < ThreadCount; i++)
                {
                    var i_local = i;
                    var workload = workloads[i_local % workloads.Length];
                    Threads[i] = new Thread(() => Exceptions[i_local] = ThreadCore(workload, i_local));
                }
            }

            //run all threads
            for (int i = 0; i < ThreadCount; i++) Threads[i].Start();
            //wait for threads to be started and ready
            for (int i = 0; i < ThreadCount; i++) barrier_threadstarted.Wait();

            //signal threads to start
            barrier_corestart.Set();

            //wait for threads to finish
            for (int i = 0; i < ThreadCount; i++) done_barrier2.Wait();

            //handle fails
            for (int i = 0; i < ThreadCount; i++)
                if (Exceptions[i] != null)
                    throw new Exception($"Thread {i} has failed: ", Exceptions[i]);

            //checks after ended
            PostRun?.Invoke(this);
        }

        public void Dispose()
        {
            barrier_threadstarted.Dispose();
            barrier_corestart.Dispose();
            done_barrier2.Dispose();
        }
    }
}