﻿/*****************************************************************************
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

using ICSharpCode.SharpZipLib.Core;
using ICSharpCode.SharpZipLib.GZip;
using ICSharpCode.SharpZipLib.Tar;
using System;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace Tensorflow.Keras.Utils
{
    public class Compress
    {
        public static void ExtractGZip(string gzipFileName, string targetDir)
        {
            // Use a 4K buffer. Any larger is a waste.    
            byte[] dataBuffer = new byte[4096];

            using (System.IO.Stream fs = new FileStream(gzipFileName, FileMode.Open, FileAccess.Read))
            {
                using (GZipInputStream gzipStream = new GZipInputStream(fs))
                {
                    // Change this to your needs
                    string fnOut = Path.Combine(targetDir, Path.GetFileNameWithoutExtension(gzipFileName));

                    using (FileStream fsOut = File.Create(fnOut))
                    {
                        StreamUtils.Copy(gzipStream, fsOut, dataBuffer);
                    }
                }
            }
        }

        public static void UnZip(String gzArchiveName, String destFolder)
        {
            var flag = gzArchiveName.Split(Path.DirectorySeparatorChar).Last().Split('.').First() + ".bin";
            if (File.Exists(Path.Combine(destFolder, flag))) return;

            Binding.tf_output_redirect.WriteLine($"Extracting.");
            var task = Task.Run(() =>
            {
                ZipFile.ExtractToDirectory(gzArchiveName, destFolder);
            });

            while (!task.IsCompleted)
            {
                Thread.Sleep(200);
                Binding.tf_output_redirect.Write(".");
            }

            File.Create(Path.Combine(destFolder, flag));
            Binding.tf_output_redirect.WriteLine("");
            Binding.tf_output_redirect.WriteLine("Extracting is completed.");
        }

        public static void ExtractTGZ(String gzArchiveName, String destFolder)
        {
            var flag = gzArchiveName.Split(Path.DirectorySeparatorChar).Last().Split('.').First() + ".bin";
            if (File.Exists(Path.Combine(destFolder, flag))) return;

            Binding.tf_output_redirect.WriteLine($"Extracting.");
            var task = Task.Run(() =>
            {
                using (var inStream = File.OpenRead(gzArchiveName))
                {
                    using (var gzipStream = new GZipInputStream(inStream))
                    {
                        using (TarArchive tarArchive = TarArchive.CreateInputTarArchive(gzipStream))
                            tarArchive.ExtractContents(destFolder);
                    }
                }
            });

            while (!task.IsCompleted)
            {
                Thread.Sleep(200);
                Binding.tf_output_redirect.Write(".");
            }

            File.Create(Path.Combine(destFolder, flag));
            Binding.tf_output_redirect.WriteLine("");
            Binding.tf_output_redirect.WriteLine("Extracting is completed.");
        }
    }
}
