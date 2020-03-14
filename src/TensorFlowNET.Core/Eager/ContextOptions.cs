using System;
using System.IO;

namespace Tensorflow.Eager
{
    public class ContextOptions : DisposableObject
    {
        public ContextOptions() : base(c_api.TFE_NewContextOptions())
        { }

        /// <summary>
        ///     Dispose any unmanaged resources related to given <paramref name="handle"/>.
        /// </summary>
        protected sealed override void DisposeUnmanagedResources(IntPtr handle) 
            => c_api.TFE_DeleteContextOptions(_handle);


        public static implicit operator IntPtr(ContextOptions opts) 
            => opts._handle;

        public static implicit operator TFE_ContextOptions(ContextOptions opts)
            => new TFE_ContextOptions(opts._handle);
        
    }

}
