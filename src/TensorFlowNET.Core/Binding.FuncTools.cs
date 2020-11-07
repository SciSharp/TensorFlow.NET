using System;

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
        }

        public class PartialFunc<Tin, Tout>
        {
            public Tin args { get; set; }
            public object[] keywords { get; set; }

            public Func<Tin, Tout> invoke { get; set; }
        }
    }
}
