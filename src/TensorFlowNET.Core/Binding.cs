using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public static partial class Binding
    {
        public static tensorflow tf { get; } = New<tensorflow>();
    }
}
