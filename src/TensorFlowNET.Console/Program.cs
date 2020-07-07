using System;
using static Tensorflow.Binding;

namespace Tensorflow
{
    class Program
    {
        static void Main(string[] args)
        {
            // boot .net core 10.5M.
            var mm = new MemoryMonitor();
            // warm up tensorflow.net 28.5M.
            mm.WarmUp();
            var cases = new MemoryTestingCases();

            int batchSize = 1000;

            // 1 million float tensor 68M.
            mm.Execute(10, 100 * batchSize, cases.Constant);

            // 100K float variable 84M.
            mm.Execute(10, 10 * batchSize, cases.Variable);

            // 1 million math add 39M.
            mm.Execute(10, 100 * batchSize, cases.MathAdd);

            // 100K gradient 44M.
            mm.Execute(10, 10 * batchSize, cases.Gradient);

            // 95M
            Console.WriteLine("Finished.");
            Console.ReadLine();
        }
    }
}
