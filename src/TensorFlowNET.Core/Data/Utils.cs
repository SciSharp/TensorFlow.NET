using System;
using System.IO;
using System.IO.Compression;
using System.Net;
using System.Threading;
using System.Threading.Tasks;

namespace Tensorflow
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

        public static async Task DownloadAsync<TDataSet>(this IModelLoader<TDataSet> modelLoader, string url, string dirSaveTo, string fileName, bool showProgressInConsole = false)
            where TDataSet : IDataSet
        {
            if (!Path.IsPathRooted(dirSaveTo))
                dirSaveTo = Path.Combine(AppContext.BaseDirectory, dirSaveTo);

            var fileSaveTo = Path.Combine(dirSaveTo, fileName);

            if (showProgressInConsole)
            {
                Binding.tf_output_redirect.WriteLine($"Downloading {fileName}");
            }

            if (File.Exists(fileSaveTo))
            {
                if (showProgressInConsole)
                {
                    Binding.tf_output_redirect.WriteLine($"The file {fileName} already exists");
                }

                return;
            }

            Directory.CreateDirectory(dirSaveTo);

            using (var wc = new WebClient())
            {
                await wc.DownloadFileTaskAsync(url, fileSaveTo).ConfigureAwait(false);
            }

        }

        public static async Task UnzipAsync<TDataSet>(this IModelLoader<TDataSet> modelLoader, string zipFile, string saveTo, bool showProgressInConsole = false)
            where TDataSet : IDataSet
        {
            if (!Path.IsPathRooted(saveTo))
                saveTo = Path.Combine(AppContext.BaseDirectory, saveTo);

            Directory.CreateDirectory(saveTo);

            if (!Path.IsPathRooted(zipFile))
                zipFile = Path.Combine(AppContext.BaseDirectory, zipFile);

            var destFileName = Path.GetFileNameWithoutExtension(zipFile);
            var destFilePath = Path.Combine(saveTo, destFileName);

            if (showProgressInConsole)
                Binding.tf_output_redirect.WriteLine($"Unzippinng {Path.GetFileName(zipFile)}");

            if (File.Exists(destFilePath))
            {
                if (showProgressInConsole)
                    Binding.tf_output_redirect.WriteLine($"The file {destFileName} already exists");
            }

            using (GZipStream unzipStream = new GZipStream(File.OpenRead(zipFile), CompressionMode.Decompress))
            {
                using (var destStream = File.Create(destFilePath))
                {
                    await unzipStream.CopyToAsync(destStream).ConfigureAwait(false);
                    await destStream.FlushAsync().ConfigureAwait(false);
                    destStream.Close();
                }

                unzipStream.Close();
            }
        }

        public static async Task ShowProgressInConsole(this Task task, bool enable)
        {
            if (!enable)
            {
                await task;
                return;
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

            await showProgressTask;
            Binding.tf_output_redirect.WriteLine("Done.");
        }

        private static async Task ShowProgressInConsole(CancellationTokenSource cts)
        {
            var cols = 0;

            await Task.Delay(100);

            while (!cts.IsCancellationRequested)
            {
                await Task.Delay(100);
                Binding.tf_output_redirect.Write(".");
                cols++;

                if (cols % 50 == 0)
                {
                    Binding.tf_output_redirect.WriteLine();
                }
            }

            if (cols > 0)
                Binding.tf_output_redirect.WriteLine();
        }
    }
}
