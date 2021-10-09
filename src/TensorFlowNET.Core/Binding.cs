using System.Diagnostics;

namespace Tensorflow
{
    public static partial class Binding
    {
        public static tensorflow tf { get; } = new tensorflow();

        /// <summary>
        ///     Alias to null, similar to python's None.
        ///     For Shape, please use Unknown
        /// </summary>
        public static readonly object None = null;

        /// <summary>
        /// Used for Shape None
        /// </summary>
        /// 
        public static readonly int Unknown = -1;
    }
}
