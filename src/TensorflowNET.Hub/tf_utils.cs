using System;
using System.IO;

namespace Tensorflow.Hub
{
    internal class tf_utils
    {
        public static string bytes_to_readable_str(long? numBytes, bool includeB = false)
        {
            if (numBytes == null) return numBytes.ToString();

            var num = (double)numBytes;

            if (num < 1024)
            {
                return $"{(long)num}{(includeB ? "B" : "")}";
            }

            num /= 1 << 10;
            if (num < 1024)
            {
                return $"{num:F2}k{(includeB ? "B" : "")}";
            }

            num /= 1 << 10;
            if (num < 1024)
            {
                return $"{num:F2}M{(includeB ? "B" : "")}";
            }

            num /= 1 << 10;
            return $"{num:F2}G{(includeB ? "B" : "")}";
        }

        public static void atomic_write_string_to_file(string filename, string contents, bool overwrite)
        {
            var tempPath = $"{filename}.tmp.{Guid.NewGuid():N}";

            using (var fileStream = new FileStream(tempPath, FileMode.Create))
            {
                using (var writer = new StreamWriter(fileStream))
                {
                    writer.Write(contents);
                    writer.Flush();
                }
            }

            try
            {
                if (File.Exists(filename))
                {
                    if (overwrite)
                    {
                        File.Delete(filename);
                        File.Move(tempPath, filename);
                    }
                }
                else
                {
                    File.Move(tempPath, filename);
                }
            }
            catch
            {
                File.Delete(tempPath);
                throw;
            }
        }

        public static string absolute_path(string path)
        {
            if (path.Contains("://"))
            {
                return path;
            }

            return Path.GetFullPath(path);
        }
    }
}
