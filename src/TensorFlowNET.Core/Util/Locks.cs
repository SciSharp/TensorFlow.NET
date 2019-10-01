using System.Threading;

namespace Tensorflow.Util
{
    /// <summary>
    ///     Provides a set of locks on different shared levels.
    /// </summary>
    public static class Locks
    {
        private static readonly ThreadLocal<object> _lockpool = new ThreadLocal<object>(() => new object());

        /// <summary>
        ///     A seperate lock for every requesting thread.
        /// </summary>
        /// <remarks>This property is thread-safe.</remarks>
        public static object ThreadWide => _lockpool.Value;


        public static readonly object ProcessWide = new object();
    }
}