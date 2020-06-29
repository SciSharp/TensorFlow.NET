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
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace Tensorflow.Util
{
    internal static class SafeHandleExtensions
    {
        /// <summary>
        /// Acquires a lease on a safe handle. The lease increments the reference count of the <see cref="SafeHandle"/>
        /// to ensure the handle is not released prior to the lease being released.
        /// </summary>
        /// <remarks>
        /// This method is intended to be used in the initializer of a <c>using</c> statement. Failing to release the
        /// lease will permanently prevent the underlying <see cref="SafeHandle"/> from being released by the garbage
        /// collector.
        /// </remarks>
        /// <param name="handle">The <see cref="SafeHandle"/> to lease.</param>
        /// <returns>A <see cref="SafeHandleLease"/>, which must be disposed to release the resource.</returns>
        /// <exception cref="ObjectDisposedException">If the lease could not be acquired.</exception>
        public static SafeHandleLease Lease(this SafeHandle handle)
        {
            if (handle is null)
                throw new ArgumentNullException(nameof(handle));

            var success = false;
            try
            {
                handle.DangerousAddRef(ref success);
                Debug.Assert(success, $"'{nameof(SafeHandle.DangerousAddRef)}' does not return when '{nameof(success)}' is false.");

                return new SafeHandleLease(handle);
            }
            catch
            {
                if (success)
                    handle.DangerousRelease();

                throw;
            }
        }
    }
}
