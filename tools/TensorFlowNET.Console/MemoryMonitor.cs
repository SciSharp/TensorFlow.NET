using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace Tensorflow
{
    public class MemoryMonitor
    {
        public void WarmUp()
        {
            var x1 = tf.Variable(10, name: "x");

            tf.compat.v1.disable_eager_execution();
            var input = np.array(4);
            var nd = tf.reshape(input, new int[] { 1, 1});
            var z = nd[0, 0];
            while (true)
            {
                var x = tf.placeholder(tf.float64, shape: (1024, 1024));
                var log = tf.log(x);

                var sess = tf.Session();
                var ones = np.ones((1024, 1024), dtype: np.float64);
                var o = sess.run(log, new FeedItem(x, ones));
                // Thread.Sleep(1);
            }

            Shape shape = (1, 32, 32, 3);
            np.arange(shape.size).astype(np.float32).reshape(shape.dims);

            print($"tensorflow native version: v{tf.VERSION}");
            tf.Context.ensure_initialized();
            var a = tf.constant(np.ones((10, 10)));
            var b = tf.Variable(a);
            var c = tf.Variable(b);
            var d = b * c;
            print(d.numpy());

            GC.Collect();
            GC.WaitForPendingFinalizers();
        }

        public void Execute(int epoch, int iterate, Action<int, int> process)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            var initialTotalMemory = Process.GetCurrentProcess().PrivateMemorySize64;
            print($"{process.Method.Name} started...");

            for (int i = 0; i < epoch; i++)
            {
                var initialMemory = Process.GetCurrentProcess().PrivateMemorySize64;
                for (int j = 0; j < iterate; j++)
                    process(i, j);

                keras.backend.clear_session();

                GC.Collect();
                GC.WaitForPendingFinalizers();
                var finalMemory = Process.GetCurrentProcess().PrivateMemorySize64;
                print($"Epoch {i}: {Format(finalMemory - initialMemory)}.");
            }

            var finalTotalMemory = Process.GetCurrentProcess().PrivateMemorySize64;
            print($"Memory usage difference: {Format(finalTotalMemory - initialTotalMemory)} / {Format(Process.GetCurrentProcess().PrivateMemorySize64)}");
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
