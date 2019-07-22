/*****************************************************************************
   Copyright 2018 The TensorFlow.NET Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

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
            int finished = 0;
            var errors = new List<string>();
            var success = new List<string>();

            var parsedArgs = ParseArgs(args);

            var examples = Assembly.GetEntryAssembly().GetTypes()
                .Where(x => x.GetInterfaces().Contains(typeof(IExample)))
                .Select(x => (IExample)Activator.CreateInstance(x))
                .Where(x => x.Enabled)
                .OrderBy(x => x.Name)
                .ToArray();

            if (parsedArgs.ContainsKey("ex"))
                examples = examples.Where(x => x.Name == parsedArgs["ex"]).ToArray();

            Console.WriteLine(Environment.OSVersion.ToString(), Color.Yellow);
            Console.WriteLine($"TensorFlow Binary v{tf.VERSION}", Color.Yellow);
            Console.WriteLine($"TensorFlow.NET v{Assembly.GetAssembly(typeof(TF_DataType)).GetName().Version}", Color.Yellow);

            for (var i = 0; i < examples.Length; i++)
                Console.WriteLine($"[{i}]: {examples[i].Name}");

            var key = "0";
            
            if (examples.Length > 1)
            {
                Console.Write($"Choose one example to run, hit [Enter] to run all: ", Color.Yellow);
                key = Console.ReadLine();
            }

            var sw = new Stopwatch();
            for (var i = 0; i < examples.Length; i++)
            {
                if (i.ToString() != key && key != "") continue;

                var example = examples[i];
                Console.WriteLine($"{DateTime.UtcNow} Starting {example.Name}", Color.White);

                try
                {
                    sw.Restart();
                    bool isSuccess = example.Run();
                    sw.Stop();

                    if (isSuccess)
                        success.Add($"Example: {example.Name} in {sw.Elapsed.TotalSeconds}s");
                    else
                        errors.Add($"Example: {example.Name} in {sw.Elapsed.TotalSeconds}s");
                }
                catch (Exception ex)
                {
                    errors.Add($"Example: {example.Name}");
                    Console.WriteLine(ex);
                }

                finished++;
                Console.WriteLine($"{DateTime.UtcNow} Completed {example.Name}", Color.White);
            }

            success.ForEach(x => Console.WriteLine($"{x} is OK!", Color.Green));
            errors.ForEach(x => Console.WriteLine($"{x} is Failed!", Color.Red));

            Console.WriteLine($"{finished} of {examples.Length} example(s) are completed.");
            Console.ReadLine();
        }

        private static Dictionary<string, string> ParseArgs(string[] args)
        {
            var parsed = new Dictionary<string, string>();

            for (int i = 0; i < args.Length; i++)
            {
                string key = args[i].Substring(1);
                switch (key)
                {
                    case "ex":
                        parsed.Add(key, args[++i]);
                        break;
                    default:
                        break;
                }
            }

            return parsed;
        }
    }
}
