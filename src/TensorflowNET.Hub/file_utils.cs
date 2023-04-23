using SharpCompress.Common;
using SharpCompress.Readers;
using System;
using System.IO;

namespace Tensorflow.Hub
{
    internal static class file_utils
    {
        //public static void extract_file(TarInputStream tgz, TarEntry tarInfo, string dstPath, uint bufferSize = 10 << 20, Action<long> logFunction = null)
        //{
        //    using (var src = tgz.GetNextEntry() == tarInfo ? tgz : null)
        //    {
        //        if (src is null)
        //        {
        //            return;
        //        }

        //        using (var dst = File.Create(dstPath))
        //        {
        //            var buffer = new byte[bufferSize];
        //            int count;

        //            while ((count = src.Read(buffer, 0, buffer.Length)) > 0)
        //            {
        //                dst.Write(buffer, 0, count);
        //                logFunction?.Invoke(count);
        //            }
        //        }
        //    }
        //}

        public static void extract_tarfile_to_destination(Stream fileobj, string dst_path, Action<long> logFunction = null)
        {
            using (IReader reader = ReaderFactory.Open(fileobj))
            {
                while (reader.MoveToNextEntry())
                {
                    if (!reader.Entry.IsDirectory)
                    {
                        reader.WriteEntryToDirectory(
                                dst_path,
                                new ExtractionOptions() { ExtractFullPath = true, Overwrite = true }
                            );
                    }
                }
            }
        }

        public static string merge_relative_path(string dstPath, string relPath)
        {
            var cleanRelPath = Path.GetFullPath(relPath).TrimStart('/', '\\');

            if (cleanRelPath == ".")
            {
                return dstPath;
            }

            if (cleanRelPath.StartsWith("..") || Path.IsPathRooted(cleanRelPath))
            {
                throw new InvalidDataException($"Relative path '{relPath}' is invalid.");
            }

            var merged = Path.Combine(dstPath, cleanRelPath);

            if (!merged.StartsWith(dstPath))
            {
                throw new InvalidDataException($"Relative path '{relPath}' is invalid. Failed to merge with '{dstPath}'.");
            }

            return merged;
        }
    }
}
