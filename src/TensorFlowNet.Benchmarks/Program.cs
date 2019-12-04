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
            if (args?.Length > 0)
            {
                for (int i = 0; i < args.Length; i++)
                {
                    string name = $"TensorFlowBenchmark.{args[i]}";
                    var type = Type.GetType(name);
                    BenchmarkRunner.Run(type);
                }
            }
            else
            {
                BenchmarkSwitcher.FromAssembly(Assembly.GetExecutingAssembly()).Run(args, ManualConfig.Create(DefaultConfig.Instance).With(ConfigOptions.DisableOptimizationsValidator));
            }

            Console.ReadLine();
        }
    }
}
