using System;
using System.IO;
using System.Collections.Generic;
using System.Net;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using NumSharp;
using SharpCompress;
using SharpCompress.Common;
using SharpCompress.Readers;

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

        public static void Unzip<TDataSet>(this IModelLoader<TDataSet> modelLoader, string zipFile, string saveTo)
            where TDataSet : IDataSet
        {
            if (!Path.IsPathRooted(saveTo))
                saveTo = Path.Combine(AppContext.BaseDirectory, saveTo);

            if (!Directory.Exists(saveTo))
                Directory.CreateDirectory(saveTo);

            using (var stream = File.OpenRead(zipFile))
            using (var reader = ReaderFactory.Open(stream))
            {
                while (reader.MoveToNextEntry())
                {
                    if (!reader.Entry.IsDirectory)
                    {
                        reader.WriteEntryToDirectory(saveTo, new ExtractionOptions()
                        {
                            ExtractFullPath = true,
                            Overwrite = true
                        });
                    }
                }
            }
        }

        public static async Task UnzipAsync<TDataSet>(this IModelLoader<TDataSet> modelLoader, string zipFile, string saveTo)
            where TDataSet : IDataSet
        {
            await Task.Run(() => modelLoader.Unzip(zipFile, saveTo));
        }

        public static async Task ShowProgressInConsole(this Task task)
        {
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
