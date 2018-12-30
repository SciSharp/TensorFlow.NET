using System;
using System.Linq;
using System.Reflection;

namespace TensorFlowNET.Examples
{
    class Program
    {
        static void Main(string[] args)
        {
            var assembly = Assembly.GetEntryAssembly();
            foreach(Type type in assembly.GetTypes().Where(x => x.GetInterfaces().Contains(typeof(IExample))))
            {
                var example = (IExample)Activator.CreateInstance(type);

                try
                {
                    example.Run();
                }
                catch (Exception ex)
                {
                    Console.WriteLine(ex);
                    Console.ReadLine();
                }
            }
        }
    }
}
