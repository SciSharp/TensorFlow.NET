using System.IO.Compression;
using System.IO;
using System;

namespace Tensorflow.NumPy;

public class NpzFormat
{
    public static void Save(NDArray[] arrays, Stream stream, CompressionLevel compression = CompressionLevel.NoCompression, bool leaveOpen = false)
    {
        using var zip = new ZipArchive(stream, ZipArchiveMode.Create, leaveOpen: leaveOpen);
        for (int i = 0; i < arrays.Length; i++)
        {
            var entry = zip.CreateEntry($"arr_{i}", compression);
            NpyFormat.Save(arrays[i], entry.Open(), leaveOpen);
        }
    }

    public static void Save(object arrays, Stream stream, CompressionLevel compression = CompressionLevel.NoCompression, bool leaveOpen = false)
    {
        var properties = arrays.GetType().GetProperties();
        using var zip = new ZipArchive(stream, ZipArchiveMode.Create, leaveOpen: leaveOpen);
        for (int i = 0; i < properties.Length; i++)
        {
            var entry = zip.CreateEntry(properties[i].Name, compression);
            var value = properties[i].GetValue(arrays);
            if (value is NDArray nd)
            {
                NpyFormat.Save(nd, entry.Open(), leaveOpen);
            }
            else
            {
                throw new NotSupportedException("Please pass in NDArray.");
            }
        }
    }
}
