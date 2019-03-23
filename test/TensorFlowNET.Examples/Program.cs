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

            foreach (Type type in assembly.GetTypes().Where(x => x.GetInterfaces().Contains(typeof(IExample))))
            {
                if (args.Length > 0 && !args.Contains(type.Name))
                    continue;

                Console.WriteLine($"{DateTime.UtcNow} Starting {type.Name}", Color.Tan);

                var example = (IExample)Activator.CreateInstance(type);

                try
                {
                    if (example.Enabled)
                        if (example.Run())
                            success.Add(type.Name);
                        else
                            errors.Add(type.Name);
                    else
                        disabled.Add(type.Name);
                }
                catch (Exception ex)
                {
                    Console.WriteLine(ex);
                }

                Console.WriteLine($"{DateTime.UtcNow} Completed {type.Name}", Color.Tan);
            }

            success.ForEach(x => Console.WriteLine($"{x} example is OK!", Color.Green));
            disabled.ForEach(x => Console.WriteLine($"{x} example is Disabled!", Color.Tan));
            errors.ForEach(x => Console.WriteLine($"{x} example is Failed!", Color.Red));

            Console.ReadLine();
        }
    }
}
