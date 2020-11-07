using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Dynamic;
using System.Text;

namespace Tensorflow
{
    public static partial class Binding
    {
        [DebuggerNonUserCode]
        public static tensorflow tf { get; } = New<tensorflow>();

        /// <summary>
        ///     Alias to null, similar to python's None.
        ///     For TensorShape, please use Unknown
        /// </summary>
        public static readonly object None = null;

        /// <summary>
        /// Used for TensorShape None
        /// </summary>
        /// 
        public static readonly int Unknown = -1;
    }
}
