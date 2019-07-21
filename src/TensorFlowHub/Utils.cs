using System;
using System.IO;
using System.Collections.Generic;
using System.Net;
using System.Text;
using System.Threading.Tasks;
using NumSharp;

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
    }
}
