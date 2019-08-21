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
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    /// <summary>
    /// Abstract class for disposable object allocated in unmanaged runtime.
    /// </summary>
    public abstract class DisposableObject : IDisposable
    {
        protected IntPtr _handle;

        protected DisposableObject() { }

        protected DisposableObject(IntPtr handle) 
            => _handle = handle;

        private void internal_dispose(bool disposing)
        {
            // free unmanaged resources (unmanaged objects).
            if (_handle != IntPtr.Zero)
            {
                // set large fields to null.
                DisposeUnmanagedResources(_handle);

                _handle = IntPtr.Zero;
            }

            if (disposing)
            {
                // dispose managed state (managed objects).
                DisposeManagedResources();
            }
        }

        /// <summary>
        ///     Dispose any managed resources.
        /// </summary>
        /// <remarks>Equivalent to what you would perform inside <see cref="Dispose()"/></remarks>
        protected virtual void DisposeManagedResources()
        {
        }

        /// <summary>
        ///     Dispose any unmanaged resources related to given <paramref name="handle"/>.
        /// </summary>
        protected abstract void DisposeUnmanagedResources(IntPtr handle);

        // override a finalizer only if Dispose(bool disposing) above has code to free unmanaged resources.
        ~DisposableObject()
        {
            // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
            internal_dispose(false);
        }

        // This code added to correctly implement the disposable pattern.
        public void Dispose()
        {
            // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
            internal_dispose(true);
            // uncomment the following line if the finalizer is overridden above.
            GC.SuppressFinalize(this);
        }
    }
}