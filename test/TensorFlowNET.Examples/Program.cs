using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Reflection;
using Console = Colorful.Console;

namespace TensorFlowNET.Examples
{
    class Program
    {
        static void Main(string[] args)
        {
            var assembly = Assembly.GetEntryAssembly();
            var errors = new List<string>();
            var success = new List<string>();
            var disabled = new List<string>();
            var examples = assembly.GetTypes()
                .Where(x => x.GetInterfaces().Contains(typeof(IExample)))
                .Select(x => (IExample)Activator.CreateInstance(x))
                .OrderBy(x => x.Priority)
                .ToArray();

            foreach (IExample example in examples)
            {
                if (args.Length > 0 && !args.Contains(example.Name))
                    continue;

                Console.WriteLine($"{DateTime.UtcNow} Starting {example.Name}", Color.White);

                try
                {
                    if (example.Enabled)
                        if (example.Run())
                            success.Add($"{example.Priority} {example.Name}");
                        else
                            errors.Add($"{example.Priority} {example.Name}");
                    else
                        disabled.Add($"{example.Priority} {example.Name}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine(ex);
                }

                Console.WriteLine($"{DateTime.UtcNow} Completed {example.Name}", Color.White);
            }

            success.ForEach(x => Console.WriteLine($"{x} example is OK!", Color.Green));
            disabled.ForEach(x => Console.WriteLine($"{x} example is Disabled!", Color.Tan));
            errors.ForEach(x => Console.WriteLine($"{x} example is Failed!", Color.Red));

            Console.ReadLine();
        }
    }
}
