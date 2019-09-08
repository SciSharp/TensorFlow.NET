using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public static partial class Binding
    {
        public static tensorflow tf { get; } = New<tensorflow>();

        /// <summary>
        ///     Alias to null, similar to python's None.
        /// </summary>
        public static readonly object None = null;
    }
}
