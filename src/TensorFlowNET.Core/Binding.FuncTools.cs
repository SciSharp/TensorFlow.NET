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
            public static PartialFunc<Tin, Tout> partial<Tin, Tout>(Func<Tin, Tout> func, Tin arg)
                => new PartialFunc<Tin, Tout>
                {
                    args = arg,
                    invoke = func
                };

            public static Func<Tin1, Tin2, Tout> partial<Tin1, Tin2, Tout>(Func<Tin1, Tin2, Tout> func, (Tin1, Tin2) args)
                => (arg1, arg2) => func(args.Item1, args.Item2);
        }

        public class PartialFunc<Tin, Tout>
        {
            public Tin args { get; set; }
            public object[] keywords { get; set; }

            public Func<Tin, Tout> invoke { get; set; }
        }
    }
}
