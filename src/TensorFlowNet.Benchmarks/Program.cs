using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Running;
using System;
using System.Reflection;
using Tensorflow.Benchmark.Leak;

namespace TensorFlowBenchmark
{
    class Program
    {
        static void Main(string[] args)
        {
            new GpuLeakByCNN().Run();

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
#pragma warning disable CS0618 // Type or member is obsolete
                BenchmarkSwitcher.FromAssembly(Assembly.GetExecutingAssembly()).Run(args, ManualConfig.Create(DefaultConfig.Instance).With(ConfigOptions.DisableOptimizationsValidator));
#pragma warning restore CS0618 // Type or member is obsolete
            }

            Console.ReadLine();
        }
    }
}
