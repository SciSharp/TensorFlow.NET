/*****************************************************************************
   Copyright 2018 The TensorFlow.NET Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

using System;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;

namespace Tensorflow
{
    /// <summary>
    /// Abstract class for disposable object allocated in unmanaged runtime.
    /// https://docs.microsoft.com/en-us/dotnet/api/system.idisposable.dispose?redirectedfrom=MSDN&view=net-5.0#System_IDisposable_Dispose
    /// </summary>
    public abstract class DisposableObject : IDisposable
    {
        protected IntPtr _handle;
        protected bool _disposed;

        [SuppressMessage("ReSharper", "UnusedMember.Global")]
        protected DisposableObject()
        { }

        protected DisposableObject(IntPtr handle)
            => _handle = handle;

        [SuppressMessage("ReSharper", "InvertIf")]
        private void Dispose(bool disposing)
        {
            if (_disposed)
                return;

            //first handle managed, they might use the unmanaged resources.
            if (disposing)
            {
                // dispose managed state (managed objects).
                DisposeManagedResources();
            }

            // free unmanaged memory
            if (_handle != IntPtr.Zero)
            {
                // Call the appropriate methods to clean up
                // unmanaged resources here.
                // If disposing is false,
                // only the following code is executed.
                DisposeUnmanagedResources(_handle);
                _handle = IntPtr.Zero;
            }

            // Note disposing has been done.
            _disposed = true;
        }

        /// <summary>
        ///     Dispose any managed resources.
        /// </summary>
        /// <remarks>Equivalent to what you would perform inside <see cref="Dispose()"/></remarks>
        protected virtual void DisposeManagedResources()
        { }

        /// <summary>
        ///     Dispose any unmanaged resources related to given <paramref name="handle"/>.
        /// </summary>
        protected abstract void DisposeUnmanagedResources(IntPtr handle);

        public void Dispose()
        {
            Dispose(true);
            // This object will be cleaned up by the Dispose method.
            // Therefore, you should call GC.SupressFinalize to
            // take this object off the finalization queue
            // and prevent finalization code for this object
            // from executing a second time.
            GC.SuppressFinalize(this);
        }

        ~DisposableObject()
        {
            Dispose(false);
        }
    }
}