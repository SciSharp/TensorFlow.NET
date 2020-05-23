using System;

namespace Tensorflow
{
    class Program
    {
        static void Main(string[] args)
        {
            // boot .net core 10.5M.
            var memoryTest = new MemoryLeakTesting();
            // warm up tensorflow.net 28.5M.
            memoryTest.WarmUp();
            // 1 million float tensor 34.5M.
            memoryTest.TensorCreation();

            Console.WriteLine("Finished.");
            Console.ReadLine();
        }
    }
}
