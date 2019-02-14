using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public static partial class tf
    {
        public static object get_collection(string key, string scope = "") => get_default_graph()
            .get_collection(key, scope: scope);
    }
}
