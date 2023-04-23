using System.IO;
using System.Threading.Tasks;

namespace Tensorflow.Hub
{
    public class GcsCompressedFileResolver : IResolver
    {
        const int LOCK_FILE_TIMEOUT_SEC = 10 * 60;
        public string Call(string handle)
        {
            var module_dir = _module_dir(handle);

            return resolver.atomic_download_async(handle, download, module_dir, LOCK_FILE_TIMEOUT_SEC)
                .GetAwaiter().GetResult();
        }
        public bool IsSupported(string handle)
        {
            return handle.StartsWith("gs://") && _is_tarfile(handle);
        }

        private async Task download(string handle, string tmp_dir)
        {
            new resolver.DownloadManager(handle).download_and_uncompress(
                new FileStream(handle, FileMode.Open, FileAccess.Read), tmp_dir);
            await Task.Run(() => { });
        }

        private static string _module_dir(string handle)
        {
            var cache_dir = resolver.tfhub_cache_dir(use_temp: true);
            var sha1 = ComputeSha1(handle);
            return resolver.create_local_module_dir(cache_dir, sha1);
        }

        private static bool _is_tarfile(string filename)
        {
            return filename.EndsWith(".tar") || filename.EndsWith(".tar.gz") || filename.EndsWith(".tgz");
        }

        private static string ComputeSha1(string s)
        {
            using (var sha = new System.Security.Cryptography.SHA1Managed())
            {
                var bytes = System.Text.Encoding.UTF8.GetBytes(s);
                var hash = sha.ComputeHash(bytes);
                var stringBuilder = new System.Text.StringBuilder(hash.Length * 2);

                foreach (var b in hash)
                {
                    stringBuilder.Append(b.ToString("x2"));
                }

                return stringBuilder.ToString();
            }
        }
    }
}
