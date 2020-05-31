using System;

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

            // 1 million float tensor 58.5M.
            mm.Execute(10, 100 * batchSize, cases.Constant);

            // 100K float variable 80.5M.
            mm.Execute(10, 10 * batchSize, cases.Variable);

            // 1 million math add 36.5M.
            mm.Execute(10, 100 * batchSize, cases.MathAdd);

            // 100K gradient 210M.
            mm.Execute(10, 10 * batchSize, cases.Gradient);

            Console.WriteLine("Finished.");
            Console.ReadLine();
        }
    }
}
