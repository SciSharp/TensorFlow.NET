using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Tensorflow
{
    public static partial class Binding
    {
        public static class functools
        {
            public static Func<Tin, Tout> partial<Tin, Tout>(Func<Tin, Tout> func, Tin arg)
                => (arg0) => func(arg0);

            public static Func<Tin1, Tin2, Tout> partial<Tin1, Tin2, Tout>(Func<Tin1, Tin2, Tout> func, (Tin1, Tin2) args)
                => (arg1, arg2) => func(arg1, arg2);
        }
    }
}
