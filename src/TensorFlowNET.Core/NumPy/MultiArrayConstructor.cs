using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Text;
using Razorvine.Pickle;

namespace Tensorflow.NumPy
{
    /// <summary>
    /// Creates multiarrays of objects. Returns a primitive type multiarray such as int[][] if 
    /// the objects are ints, etc. 
    /// </summary>
    [SuppressMessage("ReSharper", "InconsistentNaming")]
    [SuppressMessage("ReSharper", "MemberCanBePrivate.Global")]
    [SuppressMessage("ReSharper", "MemberCanBeMadeStatic.Global")]
    public class MultiArrayConstructor : IObjectConstructor
    {
        public object construct(object[] args)
        {
            //Console.WriteLine(args.Length);
            //for (int i = 0; i < args.Length; i++)
            //{
            //    Console.WriteLine(args[i]);
            //}
            Console.WriteLine("MultiArrayConstructor");

            var arg1 = (Object[])args[1];
            var dims = new int[arg1.Length];
            for (var i = 0; i < arg1.Length; i++)
            {
                dims[i] = (int)arg1[i];
            }

            var dtype = TF_DataType.DtInvalid;
            switch (args[2])
            {
                case "b": dtype = TF_DataType.DtUint8Ref; break;
                default: throw new NotImplementedException("cannot parse" + args[2]);
            }
            return new NDArray(new Shape(dims), dtype);

        }
    }
}
