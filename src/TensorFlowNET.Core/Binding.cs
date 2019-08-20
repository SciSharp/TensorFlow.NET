using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public static class Binding
    {
        public static tensorflow tf { get; } = Python.New<tensorflow>();
    }
}
