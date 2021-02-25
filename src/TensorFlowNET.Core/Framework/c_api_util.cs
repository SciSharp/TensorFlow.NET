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
            string dll = c_api.TensorFlowLibName;
            string directory = AppDomain.CurrentDomain.BaseDirectory;
            string file = "";
            string url = "";

            switch (Environment.OSVersion.Platform)
            {
                case PlatformID.Win32NT:
                    dll = $"{dll}.dll";
                    file = Path.Combine(directory, "libtensorflow-cpu-windows-x86_64-1.14.0.zip");
                    url = "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-windows-x86_64-1.14.0.zip";
                    break;
                case PlatformID.Unix:
                    dll = $"lib{dll}.so";
                    file = Path.Combine(directory, "libtensorflow-cpu-linux-x86_64-1.14.0.tar.gz");
                    url = "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.14.0.tar.gz";
                    break;
                default:
                    throw new RuntimeError($"Unknown OS environment: {Environment.OSVersion.Platform}");
            }

            if (isDllDownloaded || File.Exists($"{directory}/{dll}"))
            {
                isDllDownloaded = true;
                return;
            }

            lock (locker)
            {
                if (!File.Exists(file))
                {
                    var wc = new WebClient();
                    Binding.tf_output_redirect.WriteLine($"Downloading Tensorflow library from {url}...");
                    var download = Task.Run(() => wc.DownloadFile(url, file));
                    while (!download.IsCompleted)
                    {
                        Thread.Sleep(1000);
                        Binding.tf_output_redirect.Write(".");
                    }
                    Binding.tf_output_redirect.WriteLine("");
                    Binding.tf_output_redirect.WriteLine($"Downloaded successfully.");
                }

                Binding.tf_output_redirect.WriteLine($"Extracting...");
                var task = Task.Run(() =>
                {
                    switch (Environment.OSVersion.Platform)
                    {
                        case PlatformID.Win32NT:
                            ZipFile.ExtractToDirectory(file, directory);
                            Util.CmdHelper.Command($"move lib\\* .\\");
                            Util.CmdHelper.Command($"rm -r lib");
                            Util.CmdHelper.Command($"rm -r include");
                            break;
                        case PlatformID.Unix:
                            Util.CmdHelper.Bash($"tar xvzf {file} ./lib/");
                            Util.CmdHelper.Bash($"mv {directory}/lib/* {directory}");
                            Util.CmdHelper.Bash($"rm -r {directory}/lib");
                            break;
                        default:
                            throw new RuntimeError($"Unknown OS environment: {Environment.OSVersion.Platform}");
                    }
                });

                while (!task.IsCompleted)
                {
                    Thread.Sleep(100);
                    Binding.tf_output_redirect.Write(".");
                }

                Binding.tf_output_redirect.WriteLine("");
                Binding.tf_output_redirect.WriteLine("Extraction is completed.");
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
                yield return new Operation(c_op, graph);
            }
        }
    }
}
