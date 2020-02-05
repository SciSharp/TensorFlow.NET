using System;

namespace Tensorflow.Eager
{
    public sealed class Context : IDisposable
    {
        public const int GRAPH_MODE = 0;
        public const int EAGER_MODE = 1;

        public int default_execution_mode;
        public string device_name = "";
        public string scope_name = "";
        bool _initialized = false;

        public SafeContextHandle Handle { get; }

        public Context(ContextOptions opts, Status status)
        {
            Handle = c_api.TFE_NewContext(opts, status.Handle);
            status.Check(true);
        }

        /// <summary>
        /// Initialize handle and devices if not already done so.
        /// </summary>
        public void ensure_initialized()
        {
            if (_initialized)
                return;
            _initialized = true;
        }

        public void start_step()
            => c_api.TFE_ContextStartStep(Handle);

        public void end_step()
            => c_api.TFE_ContextEndStep(Handle);

        public bool executing_eagerly() 
            => default_execution_mode == EAGER_MODE;

        public string shared_name(string name = null)
            => !string.IsNullOrEmpty(name) || !executing_eagerly() ? 
                name : 
                "cd2c89b7-88b7-44c8-ad83-06c2a9158347";

        public void Dispose()
            => Handle.Dispose();
    }
}
