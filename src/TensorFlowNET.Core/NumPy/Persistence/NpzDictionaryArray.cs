using System.IO;
using System.IO.Compression;

namespace Tensorflow.NumPy;

public class NpzDictionary
{
    Dictionary<string, NDArray> arrays = new Dictionary<string, NDArray>();

    public NDArray this[string key] => arrays[key];

    public NpzDictionary(Stream stream)
    {
        using var archive = new ZipArchive(stream, ZipArchiveMode.Read, leaveOpen: false);

        foreach (var entry in archive.Entries)
        {
            arrays[entry.FullName] = OpenEntry(entry);
        }
    }

    private NDArray OpenEntry(ZipArchiveEntry entry)
    {
        if (arrays.TryGetValue(entry.FullName, out var array))
            return array;

        using var s = entry.Open();
        return (NDArray)LoadMatrix(s);
    }

    public Array LoadMatrix(Stream stream)
    {
        using var reader = new BinaryReader(stream, System.Text.Encoding.ASCII, leaveOpen: false);

        if (!ParseReader(reader, out var bytes, out var type, out var shape))
            throw new FormatException();

        Array matrix = Array.CreateInstance(type, shape);

        return ReadMatrix(reader, matrix, bytes, type, shape);
    }

    bool ParseReader(BinaryReader reader, out int bytes, out Type t, out int[] shape)
    {
        bytes = 0;
        t = null;
        shape = null;

        // The first 6 bytes are a magic string: exactly "x93NUMPY"
        if (reader.ReadChar() != 63) return false;
        if (reader.ReadChar() != 'N') return false;
        if (reader.ReadChar() != 'U') return false;
        if (reader.ReadChar() != 'M') return false;
        if (reader.ReadChar() != 'P') return false;
        if (reader.ReadChar() != 'Y') return false;

        byte major = reader.ReadByte(); // 1
        byte minor = reader.ReadByte(); // 0

        if (major != 1 || minor != 0)
            throw new NotSupportedException();

        ushort len = reader.ReadUInt16();

        string header = new string(reader.ReadChars(len));
        string mark = "'descr': '";
        int s = header.IndexOf(mark) + mark.Length;
        int e = header.IndexOf("'", s + 1);
        string type = header.Substring(s, e - s);
        bool? isLittleEndian;
        t = GetType(type, out bytes, out isLittleEndian);

        if (isLittleEndian.HasValue && isLittleEndian.Value == false)
            throw new Exception();

        mark = "'fortran_order': ";
        s = header.IndexOf(mark) + mark.Length;
        e = header.IndexOf(",", s + 1);
        bool fortran = bool.Parse(header.Substring(s, e - s));

        if (fortran)
            throw new Exception();

        mark = "'shape': (";
        s = header.IndexOf(mark) + mark.Length;
        e = header.IndexOf(")", s + 1);
        shape = header.Substring(s, e - s).Split(',').Where(v => !String.IsNullOrEmpty(v)).Select(Int32.Parse).ToArray();

        return true;
    }

    Type GetType(string dtype, out int bytes, out bool? isLittleEndian)
    {
        isLittleEndian = IsLittleEndian(dtype);
        bytes = int.Parse(dtype.Substring(2));

        string typeCode = dtype.Substring(1);
        return typeCode switch
        {
            "b1" => typeof(bool),
            "i1" => typeof(byte),
            "i2" => typeof(short),
            "i4" => typeof(int),
            "i8" => typeof(long),
            "u1" => typeof(byte),
            "u2" => typeof(ushort),
            "u4" => typeof(uint),
            "u8" => typeof(ulong),
            "f4" => typeof(float),
            "f8" => typeof(double),
            // typeCode.StartsWith("S") => typeof(string),
            _ => throw new NotSupportedException()
        };
    }

    bool? IsLittleEndian(string type)
    {
        return type[0] switch
        {
            '<' => true,
            '>' => false,
            '|' => null,
            _ => throw new Exception()
        };
    }

    Array ReadMatrix(BinaryReader reader, Array matrix, int bytes, Type type, int[] shape)
    {
        int total = 1;
        for (int i = 0; i < shape.Length; i++)
            total *= shape[i];

        var buffer = reader.ReadBytes(bytes * total);
        System.Buffer.BlockCopy(buffer, 0, matrix, 0, buffer.Length);

        return matrix;
    }
}
