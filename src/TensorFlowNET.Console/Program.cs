using System;
using static Tensorflow.Binding;

namespace Tensorflow
{
    class Program
    {
        static void Main(string[] args)
        {
            var diag = new Diagnostician();
            // diag.Diagnose(@"D:\memory.txt");

            // this class is used explor new features.
            var exploring = new Exploring();
            // exploring.Run();

            // boot .net core 10.5M.
            var mm = new MemoryMonitor();
            // warm up tensorflow.net 37.3M.
            mm.WarmUp();

            BasicTest(mm);

            KerasTest(mm);

            FuncGraph(mm);

            // 65M
            Console.WriteLine("Finished.");
            Console.ReadLine();
        }

        static void BasicTest(MemoryMonitor mm)
        {
            int batchSize = 1000;

            var basic = new MemoryBasicTest();

            // 1 million placeholder
            /*tf.compat.v1.disable_eager_execution();
            mm.Execute(10, 100 * batchSize, basic.Placeholder);
            tf.enable_eager_execution();*/

            // 1 million tensor
            mm.Execute(10, 100 * batchSize, basic.Constant);

            // explaination of constant
            mm.Execute(10, 100 * batchSize, basic.Constant2x3);

            mm.Execute(10, batchSize, basic.ConstantString);

            // 100K float variable.
            mm.Execute(10, batchSize, basic.Variable);

            mm.Execute(10, batchSize, basic.VariableRead);

            mm.Execute(10, batchSize, basic.VariableAssign);

            // 1 million math.
            mm.Execute(10, 100 * batchSize, basic.MathAdd);

            // Conv2d in constant tensor
            mm.Execute(10, batchSize, basic.Conv2DWithTensor);

            // Conv2d in variable
            mm.Execute(10, batchSize, basic.Conv2DWithVariable);

            // 100K gradient 44M.
            mm.Execute(10, 10 * batchSize, basic.Gradient);

            // memory leak when increasing the epoch
            mm.Execute(10, 10, basic.Dataset);
        }

        static void KerasTest(MemoryMonitor mm)
        {
            var keras = new MemoryKerasTest();

            // +1M (10,50)
            mm.Execute(10, 1, keras.Conv2DLayer);

            mm.Execute(10, 50, keras.InputLayer);

            mm.Execute(10, 10, keras.Prediction);
        }

        static void FuncGraph(MemoryMonitor mm)
        {
            var func = new MemoryFuncGraphTest();
            mm.Execute(10, 100, func.ConcreteFunction);
        }
    }
}
