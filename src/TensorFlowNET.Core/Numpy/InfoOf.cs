using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow.NumPy
{
    public class InfoOf<T>
    {
        public static readonly int Size;
        public static readonly TF_DataType NPTypeCode;
        public static readonly T Zero;
        public static readonly T MaxValue;
        public static readonly T MinValue;

        static InfoOf()
        {
            Size = NPTypeCode.get_datatype_size();
        }
    }
}
