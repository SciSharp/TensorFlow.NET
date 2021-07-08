using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow.Numpy
{
    public class InfoOf<T>
    {
        public static readonly int Size;
        public static readonly NumpyDType NPTypeCode;
        public static readonly T Zero;
        public static readonly T MaxValue;
        public static readonly T MinValue;

        static InfoOf()
        {
            NPTypeCode = typeof(T).GetTypeCode();

            switch (NPTypeCode)
            {
                case NumpyDType.Boolean:
                    Size = 1;
                    break;
                case NumpyDType.Char:
                    Size = 2;
                    break;
                case NumpyDType.Byte:
                    Size = 1;
                    break;
                case NumpyDType.Int16:
                    Size = 2;
                    break;
                case NumpyDType.UInt16:
                    Size = 2;
                    break;
                case NumpyDType.Int32:
                    Size = 4;
                    break;
                case NumpyDType.UInt32:
                    Size = 4;
                    break;
                case NumpyDType.Int64:
                    Size = 8;
                    break;
                case NumpyDType.UInt64:
                    Size = 8;
                    break;
                case NumpyDType.Single:
                    Size = 4;
                    break;
                case NumpyDType.Double:
                    Size = 8;
                    break;
                case NumpyDType.Decimal:
                    Size = 16;
                    break;
                case NumpyDType.String:
                    break;
                case NumpyDType.Complex:
                default:
                    Size = Marshal.SizeOf<T>();
                    break;
            }
        }
    }
}
