using System;
using System.IO;
using System.IO.Compression;
using System.Collections.Generic;
using System.Net;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Tensorflow.Hub
{
    public static class Utils
    {
        public static async Task DownloadAsync<TDataSet>(this IModelLoader<TDataSet> modelLoader, string url, string saveTo)
            where TDataSet : IDataSet
        {
            var dir = Path.GetDirectoryName(saveTo);
            var fileName = Path.GetFileName(saveTo);
            await modelLoader.DownloadAsync(url, dir, fileName);
        }

        public static async Task DownloadAsync<TDataSet>(this IModelLoader<TDataSet> modelLoader, string url, string dirSaveTo, string fileName)
            where TDataSet : IDataSet
        {
            if (!Path.IsPathRooted(dirSaveTo))
                dirSaveTo = Path.Combine(AppContext.BaseDirectory, dirSaveTo);

            if (!Directory.Exists(dirSaveTo))
                Directory.CreateDirectory(dirSaveTo);
            
            using (var wc = new WebClient())
            {
                await wc.DownloadFileTaskAsync(url, Path.Combine(dirSaveTo, fileName));
            }
        }

        public static async Task UnzipAsync<TDataSet>(this IModelLoader<TDataSet> modelLoader, string zipFile, string saveTo)
            where TDataSet : IDataSet
        {
            if (!Path.IsPathRooted(saveTo))
                saveTo = Path.Combine(AppContext.BaseDirectory, saveTo);

            if (!Directory.Exists(saveTo))
                Directory.CreateDirectory(saveTo);

            var destFilePath = Path.Combine(saveTo, Path.GetFileNameWithoutExtension(zipFile));

            if (File.Exists(destFilePath))
                File.Delete(destFilePath);

            using (GZipStream unzipStream = new GZipStream(File.OpenRead(zipFile), CompressionMode.Decompress))
            {
                using (var destStream = File.Create(destFilePath))
                {
                    await unzipStream.CopyToAsync(destStream);
                    await destStream.FlushAsync();
                    destStream.Close();
                }

                unzipStream.Close();
            }
        }

        public static async Task ShowProgressInConsole(this Task task)
        {
            await ShowProgressInConsole(task, true);
        }

        public static async Task ShowProgressInConsole(this Task task, bool enable)
        {
            if (!enable)
            {
                await task;
            }

            var cts = new CancellationTokenSource();
            var showProgressTask = ShowProgressInConsole(cts);
            
            try
            {
                await task;
            }
            finally
            {
                cts.Cancel();
            }            
        }

        private static async Task ShowProgressInConsole(CancellationTokenSource cts)
        {
            var cols = 0;

            while (!cts.IsCancellationRequested)
            {
                await Task.Delay(1000);
                Console.Write(".");
                cols++;

                if (cols >= 50)
                {
                    cols = 0;
                    Console.WriteLine();
                }
            }

            Console.WriteLine();
        }
    }
}
