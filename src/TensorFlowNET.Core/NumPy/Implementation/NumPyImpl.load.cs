using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using Tensorflow.Util;

namespace Tensorflow.NumPy
{
    public partial class NumPyImpl
    {
        public NDArray load(string file)
        {
            using var stream = new FileStream(file, FileMode.Open);
            using var reader = new BinaryReader(stream, Encoding.ASCII, leaveOpen: true);
            int bytes;
            Type type;
            int[] shape;
            if (!ParseReader(reader, out bytes, out type, out shape))
                throw new FormatException();

            Array array = Create(type, shape.Aggregate((dims, dim) => dims * dim));

            var result = new NDArray(ReadValueMatrix(reader, array, bytes, type, shape));
            return result.reshape(shape);
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

            string header = new String(reader.ReadChars(len));
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
            bytes = Int32.Parse(dtype.Substring(2));

            string typeCode = dtype.Substring(1);

            if (typeCode == "b1")
                return typeof(bool);
            if (typeCode == "i1")
                return typeof(Byte);
            if (typeCode == "i2")
                return typeof(Int16);
            if (typeCode == "i4")
                return typeof(Int32);
            if (typeCode == "i8")
                return typeof(Int64);
            if (typeCode == "u1")
                return typeof(Byte);
            if (typeCode == "u2")
                return typeof(UInt16);
            if (typeCode == "u4")
                return typeof(UInt32);
            if (typeCode == "u8")
                return typeof(UInt64);
            if (typeCode == "f4")
                return typeof(Single);
            if (typeCode == "f8")
                return typeof(Double);
            if (typeCode.StartsWith("S"))
                return typeof(String);

            throw new NotSupportedException();
        }

        bool? IsLittleEndian(string type)
        {
            bool? littleEndian = null;

            switch (type[0])
            {
                case '<':
                    littleEndian = true;
                    break;
                case '>':
                    littleEndian = false;
                    break;
                case '|':
                    littleEndian = null;
                    break;
                default:
                    throw new Exception();
            }

            return littleEndian;
        }

        Array Create(Type type, int length)
        {
            // ReSharper disable once PossibleNullReferenceException
            while (type.IsArray)
                type = type.GetElementType();

            return Array.CreateInstance(type, length);
        }
    }
}
