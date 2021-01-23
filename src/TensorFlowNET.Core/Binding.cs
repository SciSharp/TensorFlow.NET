using System.Diagnostics;

namespace Tensorflow
{
    public static partial class Binding
    {
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
