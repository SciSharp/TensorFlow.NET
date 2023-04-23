using System;
using System.Collections.Generic;
using System.Linq;

namespace Tensorflow.Hub
{
    internal class MultiImplRegister
    {
        private static MultiImplRegister resolver = new MultiImplRegister("resolver", new IResolver[0]);
        private static MultiImplRegister loader = new MultiImplRegister("loader", new IResolver[0]);

        static MultiImplRegister()
        {
            resolver.add_implementation(new PathResolver());
            resolver.add_implementation(new HttpUncompressedFileResolver());
            resolver.add_implementation(new GcsCompressedFileResolver());
            resolver.add_implementation(new HttpCompressedFileResolver());
        }

        string _name;
        List<IResolver> _impls;
        public MultiImplRegister(string name, IEnumerable<IResolver> impls)
        {
            _name = name;
            _impls = impls.ToList();
        }

        public void add_implementation(IResolver resolver)
        {
            _impls.Add(resolver);
        }

        public string Call(string handle)
        {
            foreach (var impl in _impls.Reverse<IResolver>())
            {
                if (impl.IsSupported(handle))
                {
                    return impl.Call(handle);
                }
            }
            throw new RuntimeError($"Cannot resolve the handle {handle}");
        }

        public static MultiImplRegister GetResolverRegister()
        {
            return resolver;
        }

        public static MultiImplRegister GetLoaderRegister()
        {
            return loader;
        }
    }
}
