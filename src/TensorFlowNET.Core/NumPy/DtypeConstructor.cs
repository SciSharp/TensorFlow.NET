using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Text;
using Razorvine.Pickle;

namespace Tensorflow.NumPy
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
            Console.WriteLine("DtypeConstructor");
            Console.WriteLine(args.Length);
            for (int i = 0; i < args.Length; i++)
            {
                Console.WriteLine(args[i]);
            }
            return new demo();
        }
    }
    class demo
    {
        public void __setstate__(object[] args)
        {
            Console.WriteLine("demo __setstate__");
            Console.WriteLine(args.Length);
            for (int i = 0; i < args.Length; i++)
            {
                Console.WriteLine(args[i]);
            }
        }
    }
}
