using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.NumPy
{
    public partial class NDArray
    {
        public void __setstate__(object[] args)
        {
            Console.WriteLine("NDArray __setstate__");
            Console.WriteLine(args.Length);
            for (int i = 0; i < args.Length; i++)
            {
                Console.WriteLine(args[i]);
            }
        }
    }
}
