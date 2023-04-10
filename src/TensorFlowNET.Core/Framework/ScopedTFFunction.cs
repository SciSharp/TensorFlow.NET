using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Framework
{
    internal class ScopedTFFunction
    {
        SafeFuncGraphHandle _handle;
        string _name;
        public ScopedTFFunction(SafeFuncGraphHandle func, string name)
        {
            _handle = func;
            _name = name;
        }

        public SafeFuncGraphHandle Get()
        {
            return _handle;
        }
    }
}
