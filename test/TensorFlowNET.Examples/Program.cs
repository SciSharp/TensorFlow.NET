using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Reflection;
using Tensorflow;
using Console = Colorful.Console;

namespace TensorFlowNET.Examples
{
    class Program
    {
        static void Main(string[] args)
        {
            var errors = new List<string>();
            var success = new List<string>();
            var disabled = new List<string>();
            var examples = Assembly.GetEntryAssembly().GetTypes()
                .Where(x => x.GetInterfaces().Contains(typeof(IExample)))
                .Select(x => (IExample)Activator.CreateInstance(x))
                .ToArray();

            Console.WriteLine($"TensorFlow v{tf.VERSION}", Color.Yellow);
            Console.WriteLine($"TensorFlow.NET v{Assembly.GetAssembly(typeof(TF_DataType)).GetName().Version}", Color.Yellow);

            var sw = new Stopwatch();
            foreach (IExample example in examples)
            {
                if (args.Length > 0 && !args.Contains(example.Name))
                    continue;

                Console.WriteLine($"{DateTime.UtcNow} Starting {example.Name}", Color.White);

                try
                {
                    if (example.Enabled || args.Length > 0) // if a specific example was specified run it, regardless of enabled value
                    {
                        sw.Restart();
                        bool isSuccess = example.Run();
                        sw.Stop();

                        if (isSuccess)
                            success.Add($"Example: {example.Name} in {sw.Elapsed.TotalSeconds}s");
                        else
                            errors.Add($"Example: {example.Name} in {sw.Elapsed.TotalSeconds}s");
                    }
                    else
                    {
                        disabled.Add($"Example: {example.Name} in {sw.ElapsedMilliseconds}ms");
                    }
                }
                catch (Exception ex)
                {
                    errors.Add($"Example: {example.Name}");
                    Console.WriteLine(ex);
                }
                
                Console.WriteLine($"{DateTime.UtcNow} Completed {example.Name}", Color.White);
            }

            success.ForEach(x => Console.WriteLine($"{x} is OK!", Color.Green));
            disabled.ForEach(x => Console.WriteLine($"{x} is Disabled!", Color.Tan));
            errors.ForEach(x => Console.WriteLine($"{x} is Failed!", Color.Red));

            Console.Write("Please [Enter] to quit.");
            Console.ReadLine();
        }
    }
}
