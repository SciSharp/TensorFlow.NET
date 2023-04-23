using System;
using System.Net.Http;
using System.Threading.Tasks;

namespace Tensorflow.Hub
{
    public class HttpCompressedFileResolver : HttpResolverBase
    {
        const int LOCK_FILE_TIMEOUT_SEC = 10 * 60; // 10 minutes

        private static readonly (string, string) _COMPRESSED_FORMAT_QUERY =
            ("tf-hub-format", "compressed");

        private static string _module_dir(string handle)
        {
            var cache_dir = resolver.tfhub_cache_dir(use_temp: true);
            var sha1 = ComputeSha1(handle);
            return resolver.create_local_module_dir(cache_dir, sha1);
        }

        public override bool IsSupported(string handle)
        {
            if (!is_http_protocol(handle))
            {
                return false;
            }
            var load_format = resolver.model_load_format();
            return load_format == Enum.GetName(typeof(resolver.ModelLoadFormat), resolver.ModelLoadFormat.COMPRESSED)
                || load_format == Enum.GetName(typeof(resolver.ModelLoadFormat), resolver.ModelLoadFormat.AUTO);
        }

        public override string Call(string handle)
        {
            var module_dir = _module_dir(handle);

            return resolver.atomic_download_async(
                    handle,
                    download,
                    module_dir,
                    LOCK_FILE_TIMEOUT_SEC
                ).GetAwaiter().GetResult();
        }

        private async Task download(string handle, string tmp_dir)
        {
            var client = new HttpClient();

            var response = await client.GetAsync(_append_compressed_format_query(handle));

            using (var httpStream = await response.Content.ReadAsStreamAsync())
            {
                new resolver.DownloadManager(handle).download_and_uncompress(httpStream, tmp_dir);
            }
        }

        private string _append_compressed_format_query(string handle)
        {
            return append_format_query(handle, _COMPRESSED_FORMAT_QUERY);
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
