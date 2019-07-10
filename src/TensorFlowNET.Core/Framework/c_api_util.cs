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
using System.IO;
using System.IO.Compression;
using System.Net;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Tensorflow
{
    public class c_api_util
    {
        static bool isDllDownloaded = false;
        static object locker = new object();
        public static void DownloadLibrary()
        {
            string dll = $"{c_api.TensorFlowLibName}.dll";

            string runtime = "win-x64";

            switch (Environment.OSVersion.Platform)
            {
                case PlatformID.Win32NT:
                    runtime = "win-x64";
                    break;
                default:
                    throw new RuntimeError($"Unknown OS environment: {Environment.OSVersion.Platform}");
            }

            if (isDllDownloaded || File.Exists(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, dll)))
            {
                isDllDownloaded = true;
                return;
            }
            
            string url = $"https://github.com/SciSharp/TensorFlow.NET/raw/master/tensorflowlib/runtimes/{runtime}/native/tensorflow.zip";

            lock (locker)
            {
                if (!File.Exists("tensorflow.zip"))
                {
                    var wc = new WebClient();
                    Console.WriteLine($"Downloading Tensorflow library...");
                    var download = Task.Run(() => wc.DownloadFile(url, Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "tensorflow.zip")));
                    while (!download.IsCompleted)
                    {
                        Thread.Sleep(1000);
                        Console.Write(".");
                    }
                    Console.WriteLine("");
                    Console.WriteLine($"Downloaded successfully.");
                }

                Console.WriteLine($"Extracting...");
                var task = Task.Run(() =>
                {
                    ZipFile.ExtractToDirectory("tensorflow.zip", AppDomain.CurrentDomain.BaseDirectory);
                });

                while (!task.IsCompleted)
                {
                    Thread.Sleep(100);
                    Console.Write(".");
                }

                Console.WriteLine("");
                Console.WriteLine("Extraction is completed.");
            }

            isDllDownloaded = true;
        }

        public static TF_Output tf_output(IntPtr c_op, int index) => new TF_Output(c_op, index);

        public static ImportGraphDefOptions ScopedTFImportGraphDefOptions() => new ImportGraphDefOptions();

        public static Buffer tf_buffer(byte[] data) => new Buffer(data);

        public static IEnumerable<Operation> new_tf_operations(Graph graph)
        {
            foreach (var c_op in tf_operations(graph))
            {
                if (graph._get_operation_by_tf_operation(c_op) == null)
                    yield return c_op;
            }
        }

        public static IEnumerable<Operation> tf_operations(Graph graph)
        {
            uint pos = 0;
            IntPtr c_op;
            while ((c_op = c_api.TF_GraphNextOperation(graph, ref pos)) != IntPtr.Zero)
            {
                yield return c_op;
            }
        }
    }
}
