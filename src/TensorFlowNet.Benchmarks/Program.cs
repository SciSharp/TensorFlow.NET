using System;
using System.Reflection;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Running;

namespace TensorFlowBenchmark
{
    class Program
    {
        static void Main(string[] args)
        {
#if DEBUG
            IConfig config = new DebugInProcessConfig();
#else
            IConfig config = null;
#endif

            if (args?.Length > 0)
            {
                for (int i = 0; i < args.Length; i++)
                {
                    string name = $"TensorFlowBenchmark.{args[i]}";
                    var type = Type.GetType(name);
                    BenchmarkRunner.Run(type, config);
                }
            }
            else
            {
                BenchmarkSwitcher.FromAssembly(Assembly.GetExecutingAssembly()).Run(args, config);
            }

            Console.ReadLine();
        }
    }
}
