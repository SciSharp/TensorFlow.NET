using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using NumSharp;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class MemoryMonitor
    {
        public void WarmUp()
        {
            print($"tensorflow native version: v{tf.VERSION}");
            var a = tf.constant(np.ones(10, 10));
            var b = tf.Variable(a);
            var c = tf.Variable(b);
            var d = b * c;
            print(d.numpy());

            GC.WaitForPendingFinalizers();
            GC.Collect();
            Thread.Sleep(1000);
        }

        public void Execute(int epoch, int iterate, Action<int> process)
        {
            print($"{process.Method.Name} started...");

            // new thread to run
            Task.Run(() =>
            {
                for (int i = 0; i < epoch; i++)
                {
                    var initialMemory = Process.GetCurrentProcess().PrivateMemorySize64;// GC.GetTotalMemory(true);
                    process(iterate);
                    var finalMemory = Process.GetCurrentProcess().PrivateMemorySize64; //GC.GetTotalMemory(true);
                    print($"Epoch {i}: {Format(finalMemory - initialMemory)}.");

                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                }
            }).Wait();

            print($"Total {process.Method.Name} usage {Format(Process.GetCurrentProcess().PrivateMemorySize64)}");
        }

        private string Format(long usage)
        {
            if (usage < 0)
                return $"-{Format(0 - usage)}";

            if (usage <= 1024 && usage >= 0)
                return $"{usage} Bytes";
            else if (usage > 1024 && usage <= 1024 * 1024)
                return $"{usage / 1024} KB";
            else
                return $"{usage / 1024 / 1024} MB";
        }
    }
}
