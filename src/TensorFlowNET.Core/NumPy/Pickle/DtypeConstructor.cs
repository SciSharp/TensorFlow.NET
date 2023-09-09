using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Text;
using Razorvine.Pickle;

namespace Tensorflow.NumPy.Pickle
{
    /// <summary>
    /// 
    /// </summary>
    [SuppressMessage("ReSharper", "InconsistentNaming")]
    [SuppressMessage("ReSharper", "MemberCanBePrivate.Global")]
    [SuppressMessage("ReSharper", "MemberCanBeMadeStatic.Global")]
    class DtypeConstructor : IObjectConstructor
    {
        public object construct(object[] args)
        {
            var typeCode = (string)args[0];
            TF_DataType dtype;
            if (typeCode == "b1")
                dtype = np.@bool;
            else if (typeCode == "i1")
                dtype = np.@byte;
            else if (typeCode == "i2")
                dtype = np.int16;
            else if (typeCode == "i4")
                dtype = np.int32;
            else if (typeCode == "i8")
                dtype = np.int64;
            else if (typeCode == "u1")
                dtype = np.ubyte;
            else if (typeCode == "u2")
                dtype = np.uint16;
            else if (typeCode == "u4")
                dtype = np.uint32;
            else if (typeCode == "u8")
                dtype = np.uint64;
            else if (typeCode == "f4")
                dtype = np.float32;
            else if (typeCode == "f8")
                dtype = np.float64;
            else if (typeCode.StartsWith("S"))
                dtype = np.@string;
            else if (typeCode.StartsWith("O"))
                dtype = np.@object;
            else
                throw new NotSupportedException();
            return new DTypePickleWarpper(dtype);
        }
    }
}
