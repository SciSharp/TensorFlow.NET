using System;
using System.Reflection;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Running;

namespace TensorFlowNet.Benchmark
{
    class Program
    {
        /// <summary>
        /// dotnet NumSharp.Benchmark.dll (Benchmark Class Name)
        /// dotnet NumSharp.Benchmark.dll nparange
        /// </summary>
        /// <param name="args"></param>
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
                    string name = $"TensorFlowNet.Benchmark.{args[i]}";
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
