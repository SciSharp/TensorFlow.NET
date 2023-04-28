using System.IO;
using System.Runtime.InteropServices;

namespace Tensorflow.NumPy;

public class NpyFormat
{
    public static void Save(NDArray array, Stream stream, bool leaveOpen = true)
    {
        using var writer = new BinaryWriter(stream, Encoding.ASCII, leaveOpen: leaveOpen);

        string dtype = GetDtypeName(array, out var type, out var maxLength);
        int[] shape = array.shape.as_int_list();
        var bytesWritten = (ulong)writeHeader(writer, dtype, shape);
        stream.Write(array.ToByteArray(), 0, (int)array.bytesize);
    }

    private static int writeHeader(BinaryWriter writer, string dtype, int[] shape)
    {
        // The first 6 bytes are a magic string: exactly "x93NUMPY"

        char[] magic = { 'N', 'U', 'M', 'P', 'Y' };
        writer.Write((byte)147);
        writer.Write(magic);
        writer.Write((byte)1); // major
        writer.Write((byte)0); // minor;

        string tuple = shape.Length == 1 ? $"{shape[0]}," : String.Join(", ", shape.Select(i => i.ToString()).ToArray());
        string header = "{{'descr': '{0}', 'fortran_order': False, 'shape': ({1}), }}";
        header = string.Format(header, dtype, tuple);
        int preamble = 10; // magic string (6) + 4

        int len = header.Length + 1; // the 1 is to account for the missing \n at the end
        int headerSize = len + preamble;

        int pad = 16 - (headerSize % 16);
        header = header.PadRight(header.Length + pad);
        header += "\n";
        headerSize = header.Length + preamble;

        if (headerSize % 16 != 0)
            throw new Exception("");

        writer.Write((ushort)header.Length);
        for (int i = 0; i < header.Length; i++)
            writer.Write((byte)header[i]);

        return headerSize;
    }

    private static string GetDtypeName(NDArray array, out Type type, out int bytes)
    {
        type = array.dtype.as_system_dtype();

        bytes = 1;

        if (type == typeof(string))
        {
            throw new NotSupportedException("");
        }
        else if (type == typeof(bool))
        {
            bytes = 1;
        }
        else
        {
            bytes = Marshal.SizeOf(type);
        }

        if (type == typeof(bool))
            return "|b1";
        else if (type == typeof(byte))
            return "|u1";
        else if (type == typeof(short))
            return "<i2";
        else if (type == typeof(int))
            return "<i4";
        else if (type == typeof(long))
            return "<i8";
        else if (type == typeof(ushort))
            return "<u2";
        else if (type == typeof(uint))
            return "<u4";
        else if (type == typeof(ulong))
            return "<u8";
        else if (type == typeof(float))
            return "<f4";
        else if (type == typeof(double))
            return "<f8";
        else if (type == typeof(string))
            return "|S" + bytes;
        else
            throw new NotSupportedException();
    }
}
