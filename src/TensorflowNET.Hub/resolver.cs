using ICSharpCode.SharpZipLib.Tar;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Net.Security;
using System.Security.Authentication;
using System.Threading.Tasks;
using System.Web;
using static Tensorflow.Binding;

namespace Tensorflow.Hub
{
    internal static class resolver
    {
        public enum ModelLoadFormat
        {
            [Description("COMPRESSED")]
            COMPRESSED,
            [Description("UNCOMPRESSED")]
            UNCOMPRESSED,
            [Description("AUTO")]
            AUTO
        }
        public class DownloadManager
        {
            private readonly string _url;
            private double _last_progress_msg_print_time;
            private long _total_bytes_downloaded;
            private int _max_prog_str;

            private bool _interactive_mode()
            {
                return !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("_TFHUB_DOWNLOAD_PROGRESS"));
            }

            private void _print_download_progress_msg(string msg, bool flush = false)
            {
                if (_interactive_mode())
                {
                    // Print progress message to console overwriting previous progress
                    // message.
                    _max_prog_str = Math.Max(_max_prog_str, msg.Length);
                    Console.Write($"\r{msg.PadRight(_max_prog_str)}");
                    Console.Out.Flush();

                    //如果flush参数为true，则输出换行符减少干扰交互式界面。
                    if (flush)
                        Console.WriteLine();

                }
                else
                {
                    // Interactive progress tracking is disabled. Print progress to the
                    // standard TF log.
                    tf.Logger.Information(msg);
                }
            }

            private void _log_progress(long bytes_downloaded)
            {
                // Logs progress information about ongoing module download.

                _total_bytes_downloaded += bytes_downloaded;
                var now = DateTime.Now.Ticks / TimeSpan.TicksPerSecond;
                if (_interactive_mode() || now - _last_progress_msg_print_time > 15)
                {
                    // Print progress message every 15 secs or if interactive progress
                    // tracking is enabled.
                    _print_download_progress_msg($"Downloading {_url}:" +
                                             $"{tf_utils.bytes_to_readable_str(_total_bytes_downloaded, true)}");
                    _last_progress_msg_print_time = now;
                }
            }

            public DownloadManager(string url)
            {
                _url = url;
                _last_progress_msg_print_time = DateTime.Now.Ticks / TimeSpan.TicksPerSecond;
                _total_bytes_downloaded = 0;
                _max_prog_str = 0;
            }

            public void download_and_uncompress(Stream fileobj, string dst_path)
            {
                // Streams the content for the 'fileobj' and stores the result in dst_path.

                try
                {
                    file_utils.extract_tarfile_to_destination(fileobj, dst_path, _log_progress);
                    var total_size_str = tf_utils.bytes_to_readable_str(_total_bytes_downloaded, true);
                    _print_download_progress_msg($"Downloaded {_url}, Total size: {total_size_str}", flush: true);
                }
                catch (TarException ex)
                {
                    throw new IOException($"{_url} does not appear to be a valid module. Inner message:{ex.Message}", ex);
                }
            }
        }
        private static Dictionary<string, string> _flags = new();
        private static readonly string _TFHUB_CACHE_DIR = "TFHUB_CACHE_DIR";
        private static readonly string _TFHUB_DOWNLOAD_PROGRESS = "TFHUB_DOWNLOAD_PROGRESS";
        private static readonly string _TFHUB_MODEL_LOAD_FORMAT = "TFHUB_MODEL_LOAD_FORMAT";
        private static readonly string _TFHUB_DISABLE_CERT_VALIDATION = "TFHUB_DISABLE_CERT_VALIDATION";
        private static readonly string _TFHUB_DISABLE_CERT_VALIDATION_VALUE = "true";

        static resolver()
        {
            set_new_flag("tfhub_model_load_format", "AUTO");
            set_new_flag("tfhub_cache_dir", null);
        }

        public static string model_load_format()
        {
            return get_env_setting(_TFHUB_MODEL_LOAD_FORMAT, "tfhub_model_load_format");
        }

        public static string? get_env_setting(string env_var, string flag_name)
        {
            string value = System.Environment.GetEnvironmentVariable(env_var);
            if (string.IsNullOrEmpty(value))
            {
                if (_flags.ContainsKey(flag_name))
                {
                    return _flags[flag_name];
                }
                else
                {
                    return null;
                }
            }
            else
            {
                return value;
            }
        }

        public static string tfhub_cache_dir(string default_cache_dir = null, bool use_temp = false)
        {
            var cache_dir = get_env_setting(_TFHUB_CACHE_DIR, "tfhub_cache_dir") ?? default_cache_dir;
            if (string.IsNullOrWhiteSpace(cache_dir) && use_temp)
            {
                // Place all TF-Hub modules under <system's temp>/tfhub_modules.
                cache_dir = Path.Combine(Path.GetTempPath(), "tfhub_modules");
            }
            if (!string.IsNullOrWhiteSpace(cache_dir))
            {
                Console.WriteLine("Using {0} to cache modules.", cache_dir);
            }
            return cache_dir;
        }

        public static string create_local_module_dir(string cache_dir, string module_name)
        {
            Directory.CreateDirectory(cache_dir);
            return Path.Combine(cache_dir, module_name);
        }

        public static void set_new_flag(string name, string value)
        {
            string[] tokens = new string[] {_TFHUB_CACHE_DIR, _TFHUB_DISABLE_CERT_VALIDATION,
                _TFHUB_DISABLE_CERT_VALIDATION_VALUE, _TFHUB_DOWNLOAD_PROGRESS, _TFHUB_MODEL_LOAD_FORMAT};
            if (!tokens.Contains(name))
            {
                tf.Logger.Warning($"You are settinng a flag '{name}' that cannot be recognized. The flag you set" +
                    "may not affect anything in tensorflow.hub.");
            }
            _flags[name] = value;
        }

        public static string _merge_relative_path(string dstPath, string relPath)
        {
            return file_utils.merge_relative_path(dstPath, relPath);
        }

        public static string _module_descriptor_file(string moduleDir)
        {
            return $"{moduleDir}.descriptor.txt";
        }

        public static void _write_module_descriptor_file(string handle, string moduleDir)
        {
            var readme = _module_descriptor_file(moduleDir);
            var content = $"Module: {handle}\nDownload Time: {DateTime.Now}\nDownloader Hostname: {Environment.MachineName} (PID:{Process.GetCurrentProcess().Id})";
            tf_utils.atomic_write_string_to_file(readme, content, overwrite: true);
        }

        public static string _lock_file_contents(string task_uid)
        {
            return $"{Environment.MachineName}.{Process.GetCurrentProcess().Id}.{task_uid}";
        }

        public static string _lock_filename(string moduleDir)
        {
            return tf_utils.absolute_path(moduleDir) + ".lock";
        }

        private static string _module_dir(string lockFilename)
        {
            var path = Path.GetDirectoryName(Path.GetFullPath(lockFilename));
            if (!string.IsNullOrEmpty(path))
            {
                return Path.Combine(path, "hub_modules");
            }

            throw new Exception("Unable to resolve hub_modules directory from lock file name.");
        }

        private static string _task_uid_from_lock_file(string lockFilename)
        {
            // Returns task UID of the task that created a given lock file.
            var lockstring = File.ReadAllText(lockFilename);
            return lockstring.Split('.').Last();
        }

        private static string _temp_download_dir(string moduleDir, string taskUid)
        {
            // Returns the name of a temporary directory to download module to.
            return $"{Path.GetFullPath(moduleDir)}.{taskUid}.tmp";
        }

        private static long _dir_size(string directory)
        {
            // Returns total size (in bytes) of the given 'directory'.
            long size = 0;
            foreach (var elem in Directory.EnumerateFileSystemEntries(directory))
            {
                var stat = new FileInfo(elem);
                size += stat.Length;
                if ((stat.Attributes & FileAttributes.Directory) != 0)
                    size += _dir_size(stat.FullName);
            }
            return size;
        }

        public static long _locked_tmp_dir_size(string lockFilename)
        {
            //Returns the size of the temp dir pointed to by the given lock file.
            var taskUid = _task_uid_from_lock_file(lockFilename);
            try
            {
                return _dir_size(_temp_download_dir(_module_dir(lockFilename), taskUid));
            }
            catch (DirectoryNotFoundException)
            {
                return 0;
            }
        }

        private static void _wait_for_lock_to_disappear(string handle, string lockFile, double lockFileTimeoutSec)
        {
            long? lockedTmpDirSize = null;
            var lockedTmpDirSizeCheckTime = DateTime.Now;
            var lockFileContent = "";

            while (File.Exists(lockFile))
            {
                try
                {
                    Console.WriteLine($"Module '{handle}' already being downloaded by '{File.ReadAllText(lockFile)}'. Waiting.");

                    if ((DateTime.Now - lockedTmpDirSizeCheckTime).TotalSeconds > lockFileTimeoutSec)
                    {
                        var curLockedTmpDirSize = _locked_tmp_dir_size(lockFile);
                        var curLockFileContent = File.ReadAllText(lockFile);

                        if (curLockedTmpDirSize == lockedTmpDirSize && curLockFileContent == lockFileContent)
                        {
                            Console.WriteLine($"Deleting lock file {lockFile} due to inactivity.");
                            File.Delete(lockFile);
                            break;
                        }

                        lockedTmpDirSize = curLockedTmpDirSize;
                        lockedTmpDirSizeCheckTime = DateTime.Now;
                        lockFileContent = curLockFileContent;
                    }
                }
                catch (FileNotFoundException)
                {
                    // Lock file or temp directory were deleted during check. Continue
                    // to check whether download succeeded or we need to start our own
                    // download.
                }

                System.Threading.Thread.Sleep(5000);
            }
        }

        public static async Task<string> atomic_download_async(
            string handle,
            Func<string, string, Task> downloadFn,
            string moduleDir,
            int lock_file_timeout_sec = 10 * 60)
        {
            var lockFile = _lock_filename(moduleDir);
            var taskUid = Guid.NewGuid().ToString("N");
            var lockContents = _lock_file_contents(taskUid);
            var tmpDir = _temp_download_dir(moduleDir, taskUid);

            // Function to check whether model has already been downloaded.
            Func<bool> checkModuleExists = () =>
                Directory.Exists(moduleDir) &&
                Directory.EnumerateFileSystemEntries(moduleDir).Any();

            // Check whether the model has already been downloaded before locking
            // the destination path.
            if (checkModuleExists())
            {
                return moduleDir;
            }

            // Attempt to protect against cases of processes being cancelled with
            // KeyboardInterrupt by using a try/finally clause to remove the lock
            // and tmp_dir.
            while (true)
            {
                try
                {
                    tf_utils.atomic_write_string_to_file(lockFile, lockContents, false);
                    // Must test condition again, since another process could have created
                    // the module and deleted the old lock file since last test.
                    if (checkModuleExists())
                    {
                        // Lock file will be deleted in the finally-clause.
                        return moduleDir;
                    }
                    if (Directory.Exists(moduleDir))
                    {
                        Directory.Delete(moduleDir, true);
                    }
                    break;  // Proceed to downloading the module.
                }
                // These errors are believed to be permanent problems with the
                // module_dir that justify failing the download.
                catch (FileNotFoundException)
                {
                    throw;
                }
                catch (UnauthorizedAccessException)
                {
                    throw;
                }
                catch (IOException)
                {
                    throw;
                }
                // All other errors are retried.
                // TODO(b/144424849): Retrying an AlreadyExistsError from the atomic write
                // should be good enough, but see discussion about misc filesystem types.
                // TODO(b/144475403): How atomic is the overwrite=False check?
                catch (Exception)
                {
                }

                // Wait for lock file to disappear.
                _wait_for_lock_to_disappear(handle, lockFile, lock_file_timeout_sec);
                // At this point we either deleted a lock or a lock got removed by the
                // owner or another process. Perform one more iteration of the while-loop,
                // we would either terminate due tf.compat.v1.gfile.Exists(module_dir) or
                // because we would obtain a lock ourselves, or wait again for the lock to
                // disappear.
            }

            // Lock file acquired.
            tf.Logger.Information($"Downloading TF-Hub Module '{handle}'...");
            Directory.CreateDirectory(tmpDir);
            await downloadFn(handle, tmpDir);
            // Write module descriptor to capture information about which module was
            // downloaded by whom and when. The file stored at the same level as a
            // directory in order to keep the content of the 'model_dir' exactly as it
            // was define by the module publisher.
            //
            // Note: The descriptor is written purely to help the end-user to identify
            // which directory belongs to which module. The descriptor is not part of the
            // module caching protocol and no code in the TF-Hub library reads its
            // content.
            _write_module_descriptor_file(handle, moduleDir);
            try
            {
                Directory.Move(tmpDir, moduleDir);
                Console.WriteLine($"Downloaded TF-Hub Module '{handle}'.");
            }
            catch (IOException e)
            {
                Console.WriteLine(e.Message);
                Console.WriteLine($"Failed to move {tmpDir} to {moduleDir}");
                // Keep the temp directory so we will retry building vocabulary later.
            }

            // Temp directory is owned by the current process, remove it.
            try
            {
                Directory.Delete(tmpDir, true);
            }
            catch (DirectoryNotFoundException)
            {
            }

            // Lock file exists and is owned by this process.
            try
            {
                var contents = File.ReadAllText(lockFile);
                if (contents == lockContents)
                {
                    File.Delete(lockFile);
                }
            }
            catch (Exception)
            {
            }

            return moduleDir;
        }
    }
    internal interface IResolver
    {
        string Call(string handle);
        bool IsSupported(string handle);
    }

    internal class PathResolver : IResolver
    {
        public string Call(string handle)
        {
            if (!File.Exists(handle) && !Directory.Exists(handle))
            {
                throw new IOException($"{handle} does not exist in file system.");
            }
            return handle;
        }
        public bool IsSupported(string handle)
        {
            return true;
        }
    }

    public abstract class HttpResolverBase : IResolver
    {
        private readonly HttpClient httpClient;
        private SslProtocol sslProtocol;
        private RemoteCertificateValidationCallback certificateValidator;

        protected HttpResolverBase()
        {
            httpClient = new HttpClient();
            _maybe_disable_cert_validation();
        }

        public abstract string Call(string handle);
        public abstract bool IsSupported(string handle);

        protected async Task<Stream> GetLocalFileStreamAsync(string filePath)
        {
            try
            {
                var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read);
                return await Task.FromResult(fs);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to read file stream: {ex.Message}");
                return null;
            }
        }

        protected async Task<Stream> GetFileStreamAsync(string filePath)
        {
            if (!is_http_protocol(filePath))
            {
                // If filePath is not an HTTP(S) URL, delegate to a file resolver.
                return await GetLocalFileStreamAsync(filePath);
            }

            var request = new HttpRequestMessage(HttpMethod.Get, filePath);
            var response = await _call_urlopen(request);

            if (response.IsSuccessStatusCode)
            {
                return await response.Content.ReadAsStreamAsync();
            }
            else
            {
                Console.WriteLine($"Failed to fetch file stream: {response.StatusCode} - {response.ReasonPhrase}");
                return null;
            }
        }

        protected void SetUrlContext(SslProtocol protocol, RemoteCertificateValidationCallback validator)
        {
            sslProtocol = protocol;
            certificateValidator = validator;
        }

        public static string append_format_query(string handle, (string, string) formatQuery)
        {
            var parsed = new Uri(handle);

            var queryBuilder = HttpUtility.ParseQueryString(parsed.Query);
            queryBuilder.Add(formatQuery.Item1, formatQuery.Item2);

            parsed = new UriBuilder(parsed.Scheme, parsed.Host, parsed.Port, parsed.AbsolutePath,
                            "?" + queryBuilder.ToString()).Uri;

            return parsed.ToString();
        }

        protected bool is_http_protocol(string handle)
        {
            return handle.StartsWith("http://") || handle.StartsWith("https://");
        }

        protected async Task<HttpResponseMessage> _call_urlopen(HttpRequestMessage request)
        {
            if (sslProtocol != null)
            {
                var handler = new HttpClientHandler()
                {
                    SslProtocols = sslProtocol.AsEnum(),
                };
                if (certificateValidator != null)
                {
                    handler.ServerCertificateCustomValidationCallback = (x, y, z, w) =>
                    {
                        return certificateValidator(x, y, z, w);
                    };
                }

                var client = new HttpClient(handler);
                return await client.SendAsync(request);
            }
            else
            {
                return await httpClient.SendAsync(request);
            }
        }

        protected void _maybe_disable_cert_validation()
        {
            if (Environment.GetEnvironmentVariable("_TFHUB_DISABLE_CERT_VALIDATION") == "_TFHUB_DISABLE_CERT_VALIDATION_VALUE")
            {
                ServicePointManager.ServerCertificateValidationCallback = (_, _, _, _) => true;
                Console.WriteLine("Disabled certificate validation for resolving handles.");
            }
        }
    }

    public class SslProtocol
    {
        private readonly string protocolString;

        public static readonly SslProtocol Tls = new SslProtocol("TLS");
        public static readonly SslProtocol Tls11 = new SslProtocol("TLS 1.1");
        public static readonly SslProtocol Tls12 = new SslProtocol("TLS 1.2");

        private SslProtocol(string protocolString)
        {
            this.protocolString = protocolString;
        }

        public SslProtocols AsEnum()
        {
            switch (protocolString.ToUpper())
            {
                case "TLS":
                    return SslProtocols.Tls;
                case "TLS 1.1":
                    return SslProtocols.Tls11;
                case "TLS 1.2":
                    return SslProtocols.Tls12;
                default:
                    throw new ArgumentException($"Unknown SSL/TLS protocol: {protocolString}");
            }
        }
    }
}
