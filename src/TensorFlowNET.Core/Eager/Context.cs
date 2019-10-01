using System;

namespace Tensorflow.Eager
{
    public class Context : DisposableObject
    {
        public const int GRAPH_MODE = 0;
        public const int EAGER_MODE = 1;

        public int default_execution_mode;

        public Context(ContextOptions opts, Status status)
        {
            _handle = c_api.TFE_NewContext(opts, status);
            status.Check(true);
        }

        /// <summary>
        ///     Dispose any unmanaged resources related to given <paramref name="handle"/>.
        /// </summary>
        protected sealed override void DisposeUnmanagedResources(IntPtr handle) 
            => c_api.TFE_DeleteContext(_handle);


        public bool executing_eagerly() => false;

        public static implicit operator IntPtr(Context ctx) 
            => ctx._handle;
    }
}
