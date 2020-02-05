using System;

namespace Tensorflow.Eager
{
    public class Context : DisposableObject
    {
        public const int GRAPH_MODE = 0;
        public const int EAGER_MODE = 1;

        public int default_execution_mode;
        public string device_name = "";
        public string scope_name = "";
        bool _initialized = false;

        public Context(ContextOptions opts, Status status)
        {
            _handle = c_api.TFE_NewContext(opts, status.Handle);
            status.Check(true);
        }

        public void ensure_initialized()
        {
            if (_initialized)
                return;
            _initialized = true;
        }

        /// <summary>
        ///     Dispose any unmanaged resources related to given <paramref name="handle"/>.
        /// </summary>
        protected sealed override void DisposeUnmanagedResources(IntPtr handle) 
            => c_api.TFE_DeleteContext(_handle);


        public bool executing_eagerly() => true;

        public string shared_name(string name = null)
            => !string.IsNullOrEmpty(name) || !executing_eagerly() ? 
                name : 
                "cd2c89b7-88b7-44c8-ad83-06c2a9158347";

        public static implicit operator IntPtr(Context ctx) 
            => ctx._handle;

        public static implicit operator TFE_Context(Context ctx)
            => new TFE_Context(ctx._handle);
    }
}
