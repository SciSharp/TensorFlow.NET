using System;

namespace Tensorflow
{
    class Program
    {
        static void Main(string[] args)
        {
            // this class is used explor new features.
            var exploring = new Exploring();
            // exploring.Run();

            // boot .net core 10.5M.
            var mm = new MemoryMonitor();
            // warm up tensorflow.net 37.3M.
            mm.WarmUp();
            var cases = new MemoryTestingCases();

            int batchSize = 1000;

            // 1 million tensor
            mm.Execute(10, 100 * batchSize, cases.Constant);

            // explaination of constant
            mm.Execute(10, 100 * batchSize, cases.Constant2x3);

            // +0M
            mm.Execute(10, batchSize, cases.Conv2dWithTensor);

            // 100K float variable 84M.
            mm.Execute(10, batchSize, cases.Variable);

            // +45M memory leak
            mm.Execute(10, batchSize, cases.Conv2dWithVariable);

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
