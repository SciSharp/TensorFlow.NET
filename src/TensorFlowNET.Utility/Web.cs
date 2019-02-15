using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace TensorFlowNET.Utility
{
    public class Web
    {
        public static bool Download(string url, string file)
        {
            if (File.Exists(file))
            {
                Console.WriteLine($"{file} already exists.");
                return false;
            }

            var wc = new WebClient();
            Console.WriteLine($"Downloading {file}");
            var download = Task.Run(() => wc.DownloadFile(url, file));
            while (!download.IsCompleted)
            {
                Thread.Sleep(1000);
                Console.Write(".");
            }
            Console.WriteLine("");
            Console.WriteLine($"Downloaded {file}");

            return true;
        }
    }
}
