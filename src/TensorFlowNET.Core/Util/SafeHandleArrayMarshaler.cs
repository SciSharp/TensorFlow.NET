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
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;

namespace Tensorflow.Util
{
    internal sealed class SafeHandleArrayMarshaler : ICustomMarshaler
    {
        private static readonly SafeHandleArrayMarshaler Instance = new SafeHandleArrayMarshaler();

        private SafeHandleArrayMarshaler()
        {
        }

#pragma warning disable IDE0060 // Remove unused parameter (method is used implicitly)
        public static ICustomMarshaler GetInstance(string cookie)
#pragma warning restore IDE0060 // Remove unused parameter
        {
            return Instance;
        }

        public int GetNativeDataSize()
        {
            return IntPtr.Size;
        }

        [HandleProcessCorruptedStateExceptions]
        public IntPtr MarshalManagedToNative(object ManagedObj)
        {
            if (ManagedObj is null)
                return IntPtr.Zero;

            var array = (SafeHandle[])ManagedObj;
            var native = IntPtr.Zero;
            var marshaledArrayHandle = false;
            try
            {
                native = Marshal.AllocHGlobal((array.Length + 1) * IntPtr.Size);
                Marshal.WriteIntPtr(native, GCHandle.ToIntPtr(GCHandle.Alloc(array)));
                marshaledArrayHandle = true;

                var i = 0;
                var success = false;
                try
                {
                    for (i = 0; i < array.Length; i++)
                    {
                        success = false;
                        var current = array[i];
                        var currentHandle = IntPtr.Zero;
                        if (current is object)
                        {
                            current.DangerousAddRef(ref success);
                            currentHandle = current.DangerousGetHandle();
                        }

                        Marshal.WriteIntPtr(native, ofs: (i + 1) * IntPtr.Size, currentHandle);
                    }

                    return IntPtr.Add(native, IntPtr.Size);
                }
                catch
                {
                    // Clean up any handles which were leased prior to the exception
                    var total = success ? i + 1 : i;
                    for (var j = 0; j < total; j++)
                    {
                        var current = array[i];
                        if (current is object)
                            current.DangerousRelease();
                    }

                    throw;
                }
            }
            catch
            {
                if (native != IntPtr.Zero)
                {
                    if (marshaledArrayHandle)
                        GCHandle.FromIntPtr(Marshal.ReadIntPtr(native)).Free();

                    Marshal.FreeHGlobal(native);
                }

                throw;
            }
        }

        public void CleanUpNativeData(IntPtr pNativeData)
        {
            if (pNativeData == IntPtr.Zero)
                return;

            var managedHandle = GCHandle.FromIntPtr(Marshal.ReadIntPtr(pNativeData, -IntPtr.Size));
            var array = (SafeHandle[])managedHandle.Target;
            managedHandle.Free();

            for (var i = 0; i < array.Length; i++)
            {
                if (array[i] is object && !array[i].IsClosed)
                    array[i].DangerousRelease();
            }
        }

        public object MarshalNativeToManaged(IntPtr pNativeData)
        {
            throw new NotSupportedException();
        }

        public void CleanUpManagedData(object ManagedObj)
        {
            throw new NotSupportedException();
        }
    }
}
