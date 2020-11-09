using System;
using System.Diagnostics;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class MemoryMonitor
    {
        public void WarmUp()
        {
            print($"tensorflow native version: v{tf.VERSION}");
        }

        public void Execute(int epoch, int iterate, Action<int> process)
        {
            /*GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();*/

            print($"{process.Method.Name} started...");
            for (int i = 0; i < epoch; i++)
            {
                var initialMemory = Process.GetCurrentProcess().PrivateMemorySize64;// GC.GetTotalMemory(true);
                process(iterate);
                var finalMemory = Process.GetCurrentProcess().PrivateMemorySize64; //GC.GetTotalMemory(true);
                print($"Epoch {i}: {Format(finalMemory - initialMemory)}.");
            }

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

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
