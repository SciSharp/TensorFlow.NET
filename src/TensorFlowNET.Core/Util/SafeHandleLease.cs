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
using System.Runtime.InteropServices;

namespace Tensorflow.Util
{
    /// <summary>
    /// Represents a lease of a <see cref="SafeHandle"/>.
    /// </summary>
    /// <seealso cref="SafeHandleExtensions.Lease"/>
    /// <devdoc>
    /// <para>Elements in this section may be referenced by <c>&lt;inheritdoc&gt;</c> elements to provide common
    /// language in documentation remarks.</para>
    ///
    /// <usage>
    /// <para>The result of this method is only valid when the underlying handle has not been disposed. If the lifetime
    /// of the object is unclear, a lease may be used to prevent disposal while the object is in use. See
    /// <see cref="SafeHandleExtensions.Lease(SafeHandle)"/>.</para>
    /// </usage>
    /// </devdoc>
    public readonly struct SafeHandleLease : IDisposable
    {
        private readonly SafeHandle _handle;

        internal SafeHandleLease(SafeHandle handle)
            => _handle = handle;

        public void Dispose()
            => _handle?.DangerousRelease();
    }
}